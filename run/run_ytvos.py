import shutil

import alphaclip
from alphaclip import alpha_clip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
import sys
from tqdm import tqdm
warnings.filterwarnings('ignore')

import os

def save_metrics_data(metrics, save_dir):
    """
    Save raw metric data as JSON for later analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ablation_metrics.json')

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[Info] Metric data saved to: {save_path}")

def get_reference_pre_frame(evfsam, alpha_clip, tokenizer, clip, exp, imgs_sam, imgs_beit, imgs_clip, video_len,
                       resize_shape, original_size_list, clip_preprocess_mask,ref_num):
    ref_masks = []
    ref_scores = []

    clip_text = alpha_clip.tokenize([exp]).cuda()
    text_features = clip.encode_text(clip_text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()

    for ref_idx in range(ref_num):
        i = int(ref_idx * (video_len - 1) / (ref_num - 1))

        ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words,
                                               resize_shape, original_size_list)
        ref_mask = (ref_mask > 0).float()
        ref_masks.append(ref_mask)

        w1, w2 = 0.5, 0.5
        alpha = clip_preprocess_mask(ref_mask).cuda()
        image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        ref_score = w1 * ref_score + w2 * torch.matmul(image_features, text_features.transpose(0, 1))[0]
        ref_scores.append(ref_score)

    return ref_masks, ref_scores

def get_ref_frames_batch(evfsam, tokenizer,  clip, clip_preprocess_mask, exp,
                         imgs_sam, imgs_beit, imgs_clip, resize_shape, original_size_list,
                         video_len, ref_num,all_blur_scores,all_imgs_clip_tensor, video_name, exp_id, save_path,sample):

    ref_masks = []
    ref_scores = []
    sample_indices = []
    raw_sample_indices = []
    for ref_idx in range(ref_num):
        i = int(ref_idx * (video_len - 1) / (ref_num - 1))
        raw_sample_indices.append(i)
    if sample:
        sample_indices = get_ref_frames_with_indices(all_blur_scores, all_imgs_clip_tensor, exp, clip, ref_num)
        # save_sampling_visualization(all_blur_scores, raw_sample_indices, sample_indices, video_name, exp_id,
        #                             save_path)
    else:
        sample_indices = raw_sample_indices
    # batch_imgs_sam = torch.stack([imgs_sam[i] for i in sample_indices]).cuda()
    # # batch_imgs_beit = torch.stack([imgs_beit[i] for i in sample_indices]).cuda()
    # batch_imgs_beit = torch.stack(imgs_beit).cuda()
    #
    # words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
    # batch_words = words.repeat(len(imgs_beit), 1)
    #
    # with torch.no_grad():
    #     batch_ref_masks, batch_iou_scores = evfsam.inference(
    #         batch_imgs_sam,
    #         batch_imgs_beit,
    #         batch_words,
    #         resize_shape,
    #         original_size_list,
    #         num_frames=ref_num,
    #         sampled_indices=torch.tensor(sample_indices)
    #     )
    #
    # clip_text = alpha_clip.tokenize([exp]).cuda()
    # text_features = clip.encode_text(clip_text)
    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #
    # for idx, i in enumerate(sample_indices):
    #     ref_mask = batch_ref_masks[idx]
    #     ref_mask = (ref_mask > 0).float()
    #     ref_masks.append(ref_mask)
    #
    #     w1, w2 = 0.5, 0.5
    #
    #     alpha = clip_preprocess_mask(ref_mask).cuda()
    #     image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #
    #     vt_alignment = torch.matmul(image_features, text_features.transpose(0, 1))[0]
    #
    #     combined_score = w1 * ref_score + w2 * vt_alignment
    #     ref_scores.append(combined_score)
    return ref_masks, ref_scores, sample_indices,raw_sample_indices


def get_ref_frames_with_indices(all_blur_scores, all_image_features, exp, clip, ref_num=5):
    """
    ，reference frame indices
    Args:
        all_blur_scores: list of per-frame FFT scores for the full video
        all_image_features: Tensor [N, D]，cached Alpha-CLIP visual features for the full video
        exp: textexpression
        clip_model: loaded CLIP model
        clip: Alpha-CLIP module used for tokenization
    """
    video_len = len(all_blur_scores)
    chunk_size = max(1, video_len // ref_num)

    sample_indices = []
    clip_text = alpha_clip.tokenize([exp]).cuda()
    with torch.no_grad():
        text_features = clip.encode_text(clip_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
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

        combined_scores = 0.3 * norm_clip + 0.7 * norm_fft

        best_local_idx = torch.argmax(combined_scores).item()
        sample_indices.append(start_idx + best_local_idx)

    return sample_indices



def test():
    metrics = {
        'raw_clip': [], 'raw_fft': [],
        'ref_clip': [], 'ref_fft': []
    }
    # initialize DAP-Mamba
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP
    alphaclip_path = 'weights/alphaclip/ViT-L-14-336px.pt'
    clip, clip_preprocess = alphaclip.load(alphaclip_path, alpha_vision_ckpt_pth='weights/alphaclip/clip_l14_336_grit_20m_4xe.pth', device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='ytvos_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir+'/experiment', 'sample3')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = '../../dataset/ref-youtube'
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    meta_file = os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_test_videos = set(data.keys())
    test_meta_file = os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json')
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])

    # inference
    progress_bar = tqdm(enumerate(video_list),
                        total=len(video_list),
                        desc='Inference on videos',
                        unit='video', file=sys.stdout)
    for idx_, video in progress_bar:
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # input pre-process
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        all_blur_scores=[]
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

            sharpness_score = get_fft_sharpness(image_np)
            all_blur_scores.append(sharpness_score)

        all_imgs_clip_tensor = torch.stack(imgs_clip, dim=0).cuda()
        N, _, H, W = all_imgs_clip_tensor.shape
        dummy_alpha = torch.ones(N, 1, H, W).cuda()
        with torch.no_grad():
            all_image_features = clip.visual(all_imgs_clip_tensor, dummy_alpha)
            all_image_features /= all_image_features.norm(dim=-1, keepdim=True)
        # for each language
        for e in range(num_expressions):

            # make files
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # per-frame mask prediction
            ref_num=5
            # ref_masks, ref_scores=get_reference_pre_frame(evfsam,alpha_clip,tokenizer, clip, exp, imgs_sam, imgs_beit,
            # imgs_clip, video_len, resize_shape, original_size_list, clip_preprocess_mask,ref_num)
            ref_masks, ref_scores,sample_indices,raw_sample_indices=get_ref_frames_batch(evfsam, tokenizer,  clip, clip_preprocess_mask, exp,
                         imgs_sam, imgs_beit, imgs_clip, resize_shape, original_size_list,
                         video_len, ref_num,all_blur_scores,all_image_features,video_name,exp_id,save_path,sample=True)

            raw_frames_dir = os.path.join(save_path, 'raw_sampled_frames')
            refined_frames_dir = os.path.join(save_path, 'refined_sampled_frames')
            os.makedirs(raw_frames_dir, exist_ok=True)
            os.makedirs(refined_frames_dir, exist_ok=True)
            #
            # for idx in raw_sample_indices:
            #     src_img_path = os.path.join(img_folder, video_name, frames[idx] + '.jpg')
            #     dst_img_path = os.path.join(raw_frames_dir, frames[idx] + '.jpg')
            #     shutil.copy(src_img_path, dst_img_path)
            #
            # for idx in sample_indices:
            #     src_img_path = os.path.join(img_folder, video_name, frames[idx] + '.jpg')
            #     dst_img_path = os.path.join(refined_frames_dir, frames[idx] + '.jpg')
            #     shutil.copy(src_img_path, dst_img_path)

            with torch.no_grad():
                clip_text = alpha_clip.tokenize([exp]).cuda()
                text_features = clip.encode_text(clip_text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                all_clip_scores = (all_image_features @ text_features.T).squeeze()

            for r_idx in raw_sample_indices:
                metrics['raw_clip'].append(all_clip_scores[r_idx].item())
                metrics['raw_fft'].append(all_blur_scores[r_idx])

            for s_idx in sample_indices:
                metrics['ref_clip'].append(all_clip_scores[s_idx].item())
                metrics['ref_fft'].append(all_blur_scores[s_idx])
            continue


            # select reference frame with highest mask score
            best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)

            # best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1))
            best_i = sample_indices[best_ref_idx]

            # forward pass
            for i in range(best_i, video_len):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == video_len - 1:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)

            # backward pass
            for i in range(best_i, -1, -1):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == 0:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)
    print("\n" + "=" * 50)
    print("      VOS sampling strategy quantitative evaluation report (DAP-Mamba)")
    print("=" * 50)

    def get_stats(data_list):
        return np.mean(data_list), np.std(data_list)

    avg_raw_c, std_raw_c = get_stats(metrics['raw_clip'])
    avg_ref_c, std_ref_c = get_stats(metrics['ref_clip'])
    avg_raw_f, std_raw_f = get_stats(metrics['raw_fft'])
    avg_ref_f, std_ref_f = get_stats(metrics['ref_fft'])

    print(f"1. CLIP semantic alignment (Higher is better):")
    print(f"   - Raw Uniform: {avg_raw_c:.4f} ± {std_raw_c:.4f}")
    print(f"   - Refined (Ours): {avg_ref_c:.4f} ± {std_ref_c:.4f}")
    print(f"   - Improvement: {((avg_ref_c - avg_raw_c) / avg_raw_c * 100):.2f}%")

    print(f"\n2. FFT image sharpness (Higher is better):")
    print(f"   - Raw Uniform: {avg_raw_f:.2f} ± {std_raw_f:.2f}")
    print(f"   - Refined (Ours): {avg_ref_f:.2f} ± {std_ref_f:.2f}")
    print(f"   - Improvement: {((avg_ref_f - avg_raw_f) / avg_raw_f * 100):.2f}%")
    print("=" * 50)
    # plot_comparison(metrics,save_path_prefix)

    save_metrics_data(metrics, save_path_prefix)

    # plot_ablation_results_zoomed(metrics, save_path_prefix)
    # plot_merged_double_column(metrics, save_path_prefix)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test()
