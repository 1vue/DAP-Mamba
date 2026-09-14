from collections import OrderedDict
import os
from models.segment_anything.utils.transforms import ResizeLongestSide
from models.dap_mamba import DAPMambaModel
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from safetensors.torch import load_file
import cv2
from transformers import AutoConfig


def get_fft_sharpness(image_np):
    """
    Compute an image sharpness score with the fast Fourier transform.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Run FFT.
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Mask low-frequency values in the center and keep high-frequency values.
    cy, cx = h // 2, w // 2
    size = 60  # Low-frequency center mask size
    fshift[cy - size:cy + size, cx - size:cx + size] = 0

    # Compute the spectrum magnitude before transforming back to the spatial domain.
    magnitude_spectrum = np.abs(fshift)
    # Use the mean magnitude as the sharpness score.
    score = np.mean(magnitude_spectrum)
    return score

def sam_preprocess(x: np.ndarray, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                   pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024):

    # Normalize colors
    x = ResizeLongestSide(img_size).apply_image(x)
    h, w = resize_shape = x.shape[:2]
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    x = (x - pixel_mean) / pixel_std

    # Pad
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x, [resize_shape]

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

# Check model weight files.
def checkWeights(path):
    print(f"\n>> Checking file: {path}")

    if not os.path.exists(path):
        print("❌ File does not exist")
        return

    try:
        # Choose the loader from the checkpoint suffix.
        if path.endswith(".safetensors"):
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")

        # Unwrap checkpoints saved by common distributed training wrappers.
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            print("Detected a state_dict wrapper and unpacked it")
            state_dict = state_dict["state_dict"]

        print(f"Total parameter tensors: {len(state_dict)}")
        print("-" * 80)
        print(f"{'Key Name (Layer)':<55} | Shape")
        print("-" * 80)

        for i, (key, value) in enumerate(state_dict.items()):
            print(f"{key:<55} | {list(value.shape)}")

    except Exception as e:
        print(f"Failed to read checkpoint: {e}")


# Load trained model weights for inference.
def load_checkpoint(model, checkpointPath):

    state_dict = torch.load(checkpointPath, map_location='cpu',weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # Strip the 'module.' prefix saved by distributed training.
        if k.startswith('module.'):
            print("Removed 'module.' prefix")
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)

# Load merged weights before training.
def load_new_model(evfsam,path):
    print(f"   -> Loading merged model weights: {path}")
    if os.path.exists(path):
        # Read the merged weights.
        state_dict = torch.load(path, map_location='cpu')
        # A key mismatch means the layer names in code and weights are not aligned.
        missing_keys, unexpected_keys = evfsam.load_state_dict(state_dict, strict=True)
        print("   Weights loaded successfully! (strict=True check passed)")
        return True
    else:
        print(f"   Error: merged weight file not found: {path}")
        return False

def init_models():
    print("Initializing the DAP-Mamba model for inference")
    model_path = 'models/pretrainedModels/evf-sam-multitask'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', use_fast=False,local_files_only=True)
    # dap_mamba = DAPMambaModel.from_pretrained(model_path, low_cpu_mem_usage=False, local_files_only=True)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    evfsam = DAPMambaModel(config)
    pretrained_path = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(pretrained_path):
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        current_state = evfsam.state_dict()
        matched_keys = []
        for key in pretrained_state.keys():
            if key in current_state:
                current_state[key] = pretrained_state[key]
                matched_keys.append(key)
        evfsam.load_state_dict(current_state)
        print(f"Loaded {len(matched_keys)} pretrained tensors")
        print(f"Skipped {len(current_state) - len(matched_keys)} new module tensors")
    else:
        print("Pretrained weight file was not found")

    # path="weights/checkPoint_full_video_mamba/video_mamba0002.pth"
    path="weights/mevis/checkPoint_full_mamba/video_mamba0002.pth"
    load_checkpoint(evfsam, path)
    # res=load_hq_sam(evfsam, "weights/hq_sam/evfsam_hq_merged_mapped.pth")
    evfsam = evfsam.cuda()
    evfsam.eval()
    print(f"Loaded weights: {path}; model initialization is complete")
    return tokenizer, evfsam
