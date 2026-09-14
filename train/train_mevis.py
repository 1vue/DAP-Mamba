from models.dap_mamba import DAPMambaModel
from datasets.mevis import MeViSDataset
from datasets.transforms_image import ResizeLongestSide
from transformers import AutoTokenizer, BitsAndBytesConfig
from einops import rearrange, repeat
from tqdm import tqdm
import os
import torch

import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import warnings
import time
from transformers import AutoConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings('ignore')
# from utils import init_models

def printLayers(new_layer_params,old_layer_params,captured_names):

    print(f"\n📊 Parameter group summary:")
    print(f"   🔥 New layers (High LR 5e-5): {len(new_layer_params)}  tensors")
    print(f"   ❄️ Old layers (Low LR 5e-6):  {len(old_layer_params)}  tensors")
    print(f"   🔍 New layersexample (first 5):")
    for n in captured_names:
        print(f"      - {n}")
    print("-" * 50)

def getlayers(rank,model):
    new_layer_params = []
    old_layer_params = []
    target_keywords = [
        'temporal_fusion','prompt_interaction','weight_proj'
    ]
    captured_names = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in target_keywords):
            new_layer_params.append(param)
            if rank == 0: captured_names.append(name)
        else:
            old_layer_params.append(param)
    if rank == 0:
        printLayers(new_layer_params, old_layer_params, captured_names)

    return new_layer_params, old_layer_params


def check_initialization(model):
    print("=== Checking new module initialization ===")

    for name, param in model.named_parameters():
        if 'temporal_fusion' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Range: [{param.min().item():.3f}, {param.max().item():.3f}]")

            if 'out_proj.weight' in name:
                if param.std().item() > 0.1:
                    print(f"  ⚠️  Warning: out_proj weight std {param.std().item():.3f} may be too large")
                    print(f"     Recommend Xavier initialization with gain=0.01")



def init_models():
    print("Initializing DAP-Mamba model for training")
    model_path = 'models/pretrainedModels/evf-sam-multitask'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', use_fast=False, local_files_only=True)
    # dap_mamba = DAPMambaModel.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    evfsam = DAPMambaModel(config)
    pretrained_path = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(pretrained_path):
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        current_state = evfsam.state_dict()
        matched_keys = []
        skipped_keys = []
        for key in current_state.keys():
            if key in pretrained_state:
                if current_state[key].shape == pretrained_state[key].shape:
                    current_state[key] = pretrained_state[key]
                    matched_keys.append(key)
                else:
                    skipped_keys.append(
                        f"{key} (Shape Mismatch: {pretrained_state[key].shape} -> {current_state[key].shape})")
            else:
                skipped_keys.append(key)
        evfsam.load_state_dict(current_state)
        print(f"✅ Loaded {len(matched_keys)} pretrained tensors")
        print(f"⚠️  Skipped {len(skipped_keys)} new module tensors")
    else:
        print("⚠️  Pretrained weight file not found")

    evfsam = evfsam.cuda()
    evfsam.train()
    return tokenizer, evfsam

def train():

    # initialize distributed training
    if torch.cuda.device_count() > 1:
        deepspeed.init_distributed()
    ds_config = {
        'train_batch_size': 10,
        'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': 5,
        "gradient_clipping": 0.5,
        'fp16': {
            'enabled': True,
            'initial_scale_power': 8
        },
        'zero_optimization': {
            'stage': 2
        }
    }   

    # initialize models
    tokenizer, model = init_models()
    
    # load dataset
    root = '../../dataset/mevis'
    img_folder = os.path.join(root, 'train')
    ann_file = os.path.join(root, 'train', 'meta_expressions.json')
    dataset = MeViSDataset(img_folder=img_folder, ann_file=ann_file, tf=ResizeLongestSide(1024),
                           return_masks=True, num_frames=5, max_skip=3)
    
    # setup distributed training
    if torch.cuda.device_count() > 1:
        rank = deepspeed.comm.get_rank()
        world_size = deepspeed.comm.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        rank = 0
        sampler = None
    dataset_loader = DataLoader(dataset, batch_size=1, num_workers=4, sampler=sampler)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-6)

    new_layer_params, old_layer_params = getlayers(rank, model)
    if len(new_layer_params) == 0:
        base_optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    else:
        base_optimizer = optim.AdamW([
            {'params': new_layer_params, 'lr': 5e-6},
            {'params': old_layer_params, 'lr': 1e-6},
        ])
    max_epochs = 10
    # initialize deepspeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=base_optimizer,
        model_parameters=model.parameters(),
        config=ds_config
    )
    start_epoch = 1
    load_path, client_state = model.load_checkpoint(load_dir='weights/mevis/checkPoint_full_mamba',
                                                    load_module_strict=False)
    # training loop
    if load_path is not None:
        start_epoch = client_state['epoch'] + 1
        print(f"✅ Resumed training from epoch {start_epoch} ")
    else:
        print("⚠️ Checkpoint not found; training from scratch")

    scheduler = CosineAnnealingLR(base_optimizer, T_max=max_epochs, eta_min=1e-7, last_epoch=start_epoch - 2)
    # scheduler = CosineAnnealingLR(base_optimizer, T_max=max_epochs, eta_min=1e-6)
    for epoch in range(start_epoch, max_epochs + 1):
        sampler.set_epoch(epoch)
        loss_sum = 0
        bce_loss_sum = 0
        dice_loss_sum = 0
        iters_100_time = 0
        for iter, data in tqdm(enumerate(dataset_loader, 1), total=len(dataset_loader), desc=f'Epoch {epoch}/{max_epochs}'):
            iter_start_time = time.time()
            imgs_sam, imgs_beit, targets = data
            # imgs_sam = imgs_sam[:, 0, :, :, :].unsqueeze(1).cuda().half()
            # imgs_beit = imgs_beit[:, 0, :, :, :].unsqueeze(1).cuda().half()

            imgs_sam = imgs_sam.cuda().half()
            imgs_beit = imgs_beit.cuda().half()
            sampled_indices = targets['sampled_indices']

            B, T, C, H, W = imgs_sam.shape
            resize = [(H, W)]
            original_size = targets['orig_size']
            vid_len = targets['vid_len']
            # text pre-process
            exp = targets['caption'][0]
            input_ids = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
            attn_masks = torch.zeros_like(input_ids)
            input_ids = input_ids.repeat(vid_len, 1)
            attn_masks = attn_masks.repeat(vid_len, 1)
            size_tensor = targets['orig_size']
            size_tuple = tuple(size_tensor.squeeze().tolist())
            resize = targets['resize'][0]

            # image pre-process
            imgs_sam = rearrange(imgs_sam, 'b t c h w -> (b t) c h w', b=B, t=T)
            imgs_beit = rearrange(imgs_beit, 'b t c h w -> (b t) c h w', b=B, t=vid_len)
            # masks = rearrange(targets['masks'][:, 0, :, :].unsqueeze(1), 'b t h w -> (b t) h w', b=B, t=T)
            masks = rearrange(targets['masks'], 'b t h w -> (b t) h w', b=B, t=T)
            masks = masks.cuda()


            # loss
            optimizer.zero_grad()
            # with torch.amp.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            loss = model(imgs_sam, imgs_beit, input_ids, torch.ones_like(input_ids), None, masks, size_tuple, resize,num_frames=T,sampled_indices=sampled_indices)
            loss_sum += loss['loss'].item()
            bce_loss_sum += loss['mask_bce_loss'].item()
            dice_loss_sum += loss['mask_dice_loss'].item()

            iter_time = time.time() - iter_start_time
            iters_100_time += iter_time
            # print stats
            if rank == 0 and (iter + 1) % 100 == 0:
                avg_loss = loss_sum / (iter)
                avg_loss_bce = bce_loss_sum / (iter)
                avg_loss_dice = dice_loss_sum / (iter)
                if len(optimizer.param_groups) > 1:
                    new_layer_lr = optimizer.param_groups[0]['lr']
                    old_layer_lr = optimizer.param_groups[1]['lr']
                    lr_info = f"old_lr={old_layer_lr:.2e} | new_lr={new_layer_lr:.2e}"
                else:
                    old_layer_lr = optimizer.param_groups[0]['lr']
                    lr_info = f"lr={old_layer_lr:.2e}"

                print(
                    f"\n[it{iter :04d}] "
                    f"loss={avg_loss:.5f} | bce={avg_loss_bce:.5f} | dice={avg_loss_dice:.5f} | "
                    f"{lr_info} | time/iter={iters_100_time / 100:.3f}s"
                )
            iters_100_time = 0
            # backward
            model.backward(loss['loss'])
            model.step()

        # refresh stats
        loss_sum = 0
        bce_loss_sum = 0
        dice_loss_sum = 0
        scheduler.step()
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir='weights/mevis/checkPoint_full_mamba', tag=f'epoch_{epoch:04d}',
                              client_state=client_state)
        if rank == 0:
            print(f"Checkpoint saved: weights/mevis/checkPoint_full_mamba/epoch_{epoch:04d}")
            save_path = os.path.join('weights/mevis/checkPoint_full_mamba/video_mamba{:04d}.pth'.format(epoch))
            model.save_16bit_model(os.path.dirname(save_path), os.path.basename(save_path))
            print(f"💾 (DeepSpeed) Generated .pth file: {save_path}")

if __name__ == '__main__':
    torch.cuda.set_device(0)
    # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
    train()
