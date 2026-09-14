import sys
import os
import alphaclip
from alphaclip import alpha_clip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
import cv2
import json

from PIL import Image

import torchvision as tv
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')
import torch

def get_ref_frames_batch(evfsam, tokenizer,  clip, clip_preprocess_mask, exp,
                         imgs_sam, imgs_beit, imgs_clip, resize_shape, original_size_list,
                         video_len, ref_num,all_blur_scores,all_imgs_clip_tensor, video_name, exp_id, save_path,img_folder,frames,sample=True):

    ref_masks = []
    ref_scores = []
    sample_indices = []
    raw_sample_indices = []
    for ref_idx in range(ref_num):
        i = int(ref_idx * (video_len - 1) / (ref_num - 1))
        raw_sample_indices.append(i)
    if sample:
        sample_indices=get_ref_frames_with_indices(all_blur_scores,all_imgs_clip_tensor,exp,clip,ref_num)
        # save_sampling_visualization(all_blur_scores, raw_sample_indices, sample_indices, video_name, exp_id,
        #                             save_path)
    else:
        sample_indices=raw_sample_indices

    batch_imgs_sam = torch.stack([imgs_sam[i] for i in sample_indices]).cuda()
    # batch_imgs_beit = torch.stack([imgs_beit[i] for i in sample_indices]).cuda()
    batch_imgs_beit = torch.stack(imgs_beit).cuda()
    words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
    batch_words = words.repeat(len(imgs_beit), 1)

    # vis_save_dir = os.path.join(save_path, "attention_vis")

    with torch.no_grad():
        batch_ref_masks, batch_iou_scores= evfsam.inference(
            batch_imgs_sam,
            batch_imgs_beit,
            batch_words,
            resize_shape,
            original_size_list,
            num_frames=ref_num,
            sampled_indices=torch.tensor(sample_indices)
        )

    clip_text = alpha_clip.tokenize([exp]).cuda()
    text_features = clip.encode_text(clip_text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for idx, i in enumerate(sample_indices):
        ref_mask = batch_ref_masks[idx]
        ref_mask = (ref_mask > 0).float()
        ref_masks.append(ref_mask)

        w1, w2 = 0.5, 0.5
        ref_score = batch_iou_scores[idx]

        alpha = clip_preprocess_mask(ref_mask).cuda()
        image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        vt_alignment = torch.matmul(image_features, text_features.transpose(0, 1))[0]

        combined_score = w1 * ref_score + w2 * vt_alignment
        ref_scores.append(combined_score)

    return ref_masks, ref_scores, sample_indices


def get_ref_frames_with_indices(all_blur_scores, all_imgs_clip_tensor, exp, clip, ref_num=5):
    """
    ，reference frame indices
    Args:
        all_blur_scores: list of per-frame FFT scores for the full video
        all_imgs_clip_tensor: Tensor [N, 3, 336, 336]，Alpha-CLIP visual features for the full video
        exp: textexpression
        clip_model: loaded CLIP model
        clip: Alpha-CLIP module used for tokenization
    """
    video_len = len(all_blur_scores)
    chunk_size = video_len // ref_num

    sample_indices = []
    N, C, H, W = all_imgs_clip_tensor.shape
    dummy_alpha = torch.ones(
        N, 1, H, W,
        device=all_imgs_clip_tensor.device,
        dtype=all_imgs_clip_tensor.dtype
    )
    clip_text = alpha_clip.tokenize([exp]).cuda()
    with torch.no_grad():
        text_features = clip.encode_text(clip_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        all_image_features = clip.visual(all_imgs_clip_tensor, dummy_alpha)
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
    for g in range(ref_num):
        start_idx = g * chunk_size
        end_idx = (g + 1) * chunk_size if g != ref_num - 1 else video_len

        if start_idx >= end_idx:
            break

        chunk_clip_feats = all_image_features[start_idx:end_idx]  # [N, 1024]
        clip_scores = (chunk_clip_feats @ text_features.T).squeeze()  # Shape: [N]

        fft_scores = torch.tensor(all_blur_scores[start_idx:end_idx]).to(clip_scores.device)  # Shape: [N]

        def min_max_normalize(t):
            if t.max() == t.min():
                return torch.ones_like(t) * 0.5
            return (t - t.min()) / (t.max() - t.min() + 1e-8)

        norm_clip = min_max_normalize(clip_scores)
        norm_fft = min_max_normalize(fft_scores)

        combined_scores = 0.7 * norm_clip + 0.3 * norm_fft

        best_local_idx = torch.argmax(combined_scores).item()
        sample_indices.append(start_idx + best_local_idx)

    return sample_indices



def inference_davis():
    tokenizer, evfsam = init_models()
    # initialize Alpha-CLIP
    alphaclip_path = 'weights/alphaclip/ViT-L-14-336px.pt'
    clip, clip_preprocess = alphaclip.load(alphaclip_path,
                                           alpha_vision_ckpt_pth='weights/alphaclip/clip_l14_336_grit_20m_4xe.pth',
                                           device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    cutie = get_default_model(config='davis_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir+'/davis_mamba', 'Annotations_new_cuite_2')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    root = '../../dataset/davis'
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    meta_file = os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    video_list = sorted(list(data.keys()))

    palette_img = os.path.join(root, 'valid', 'Annotations', 'blackswan', '00000.png')
    if os.path.exists(palette_img):
        palette = Image.open(palette_img).getpalette()
    else:
        palette = None

    start_time = time.time()
    progress_bar = tqdm(enumerate(video_list),
                        total=len(video_list),
                        desc='Inference on videos',
                        unit='video', file=sys.stdout)
    for idx_, video in progress_bar:
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]['frames'])
        frames = data[video]['frames']

        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas

        num_obj = num_expressions // 4

        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        all_blur_scores=[]
        for i in range(video_len):
            img_path = os.path.join(img_folder, video, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

            sharpness_score = get_fft_sharpness(image_np)
            all_blur_scores.append(sharpness_score)
        all_imgs_clip_tensor = torch.stack(imgs_clip, dim=0).cuda()

        for anno_id in range(4):
            anno_masks = []  # [num_obj, video_len, h, w]

            for obj_id in range(num_obj):
                i = obj_id * 4 + anno_id
                video_name = meta[i]['video']
                exp = meta[i]['exp']
                exp_id = meta[i]['exp_id']
                frames = meta[i]['frames']

                ref_masks = []
                ref_scores = []
                ref_num = 5
                current_vis_subdir = f"anno_{anno_id}_exp_{exp_id}"
                save_path = os.path.join(save_path_prefix, video_name, current_vis_subdir)

                ref_masks, ref_scores, sample_indices = get_ref_frames_batch(evfsam, tokenizer, clip,
                                                                             clip_preprocess_mask, exp,
                                                                             imgs_sam, imgs_beit, imgs_clip,
                                                                             resize_shape, original_size_list,
                                                                             video_len, ref_num, all_blur_scores,
                                                                             all_imgs_clip_tensor, video_name, exp_id,
                                                                             save_path,img_folder,frames,sample=True)

                best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)
                # best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1)) if ref_num > 1 else 0
                best_i = sample_indices[best_ref_idx]
                obj_masks = [None] * video_len

                for frame_idx in range(best_i, video_len):
                    if frame_idx == best_i:
                        mask_prob = processor.step(imgs_cutie[frame_idx].cuda(), ref_masks[best_ref_idx].squeeze(0),
                                                   objects=[1])
                    else:
                        mask_prob = processor.step(imgs_cutie[frame_idx].cuda())
                    mask = processor.output_prob_to_mask(mask_prob).float()

                    if frame_idx == video_len - 1:
                        processor.clear_memory()

                    obj_masks[frame_idx] = mask

                for frame_idx in range(best_i, -1, -1):
                    if frame_idx == best_i:
                        mask_prob = processor.step(imgs_cutie[frame_idx].cuda(), ref_masks[best_ref_idx].squeeze(0),
                                                   objects=[1])
                    else:
                        mask_prob = processor.step(imgs_cutie[frame_idx].cuda())
                    mask = processor.output_prob_to_mask(mask_prob).float()

                    if frame_idx == 0:
                        processor.clear_memory()

                    obj_masks[frame_idx] = mask

                obj_masks_np = []
                for mask in obj_masks:
                    mask_np = mask.detach().cpu().numpy().astype(np.float32)
                    obj_masks_np.append(mask_np)
                obj_masks_np = np.stack(obj_masks_np, axis=0)  # [video_len, h, w]
                anno_masks.append(obj_masks_np)

            anno_masks = np.stack(anno_masks)  # [num_obj, video_len, h, w]
            t, h, w = anno_masks.shape[-3:]

            anno_masks[anno_masks < 0.5] = 0.0

            background = 0.1 * np.ones((1, t, h, w), dtype=np.float32)
            anno_masks = np.concatenate([background, anno_masks], axis=0)  # [num_obj+1, video_len, h, w]

            out_masks = np.argmax(anno_masks, axis=0).astype(np.uint8)  # [video_len, h, w]

            anno_save_path = os.path.join(save_path_prefix, f'anno_{anno_id}', video)
            if not os.path.exists(anno_save_path):
                os.makedirs(anno_save_path)

            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                if palette is not None:
                    img_E.putpalette(palette)
                img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))



if __name__ == '__main__':
    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        inference_davis()
