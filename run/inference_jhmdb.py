import sys
import os

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from models.propagation_processor import Sam2Processor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import json
from PIL import Image
import torchvision as tv
from pycocotools import mask as mask_util
import warnings
import argparse
warnings.filterwarnings('ignore')
import alphaclip
from alphaclip import alpha_clip

import time

def load_jhmdb_annotations(dataset_path):
    """Load JHMDB-Sentences single-frame test annotations.
    Expected JSON structure: list of [video_id, frame_path, mask_path, instance_id, text_query].
    """
    ann_file = os.path.join(dataset_path, 'jhmdb_sentences_samples_metadata.json')
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    with open(ann_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations


def get_video_dir(video_id, dataset_path):
    """Find the directory containing the frames for a given video_id."""
    frames_dir = os.path.join(dataset_path, 'Rename_Images')
    for category in os.listdir(frames_dir):
        category_path = os.path.join(frames_dir, category)
        if os.path.isdir(category_path):
            candidate_video_dir = os.path.join(category_path, video_id)
            if os.path.exists(candidate_video_dir):
                return candidate_video_dir
    return None

def safe_read_image(path):
    try:
        img = np.array(Image.open(path).convert("RGB"))
        return img
    except Exception as e:
        print(f"PIL failed: {path}, {e}")
        return None

def read_all_frames(video_id, dataset_path):
    """Read all frames (RGB) from JHMDB video directory."""
    frames_dir = os.path.join(dataset_path, 'Rename_Images')

    video_dir = None
    for category in os.listdir(frames_dir):
        category_path = os.path.join(frames_dir, category)
        if os.path.isdir(category_path):
            candidate_video_dir = os.path.join(category_path, video_id)
            if os.path.exists(candidate_video_dir):
                video_dir = candidate_video_dir
                break

    if video_dir is None:
        raise FileNotFoundError(f"Video directory not found for {video_id}")

    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])
    if not frame_files:
        raise RuntimeError(f"No frames found in {video_dir}")

    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_dir, frame_file)
        frame = safe_read_image(frame_path)
        if frame is None:
            raise RuntimeError(f"Failed to read: {frame_path}")
        frames.append(frame)

    return frames


def get_image_id(video_id, frame_idx):
    """Generate COCO-format image_id values aligned with GT files."""
    return f'v_{video_id}_f_{int(frame_idx)}'

def get_ref_frames_batch(evfsam, tokenizer,  clip, clip_preprocess_mask, exp,
                         imgs_sam, imgs_beit, imgs_clip, resize_shape, original_size_list,
                         video_len, ref_num,all_blur_scores,all_imgs_clip_tensor,sample):

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
    batch_imgs_sam = torch.stack([imgs_sam[i] for i in sample_indices]).cuda()
    # batch_imgs_beit = torch.stack([imgs_beit[i] for i in sample_indices]).cuda()
    batch_imgs_beit = torch.stack(imgs_beit).cuda()
    words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
    batch_words = words.repeat(len(imgs_beit), 1)

    with torch.no_grad():
        batch_ref_masks, batch_iou_scores = evfsam.inference(
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


import time
from datetime import timedelta


def format_elapsed_eta(start_time, done, total):
    """
    Format elapsed time and estimated remaining time.

    Args:
        start_time: timestamp
        done: completed count
        total: total count

    Returns:
        elapsed_str: elapsed time string
        eta_str: ETA string
    """
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    if done > 0:
        eta = (elapsed / done) * (total - done)
        eta_str = str(timedelta(seconds=int(eta)))
    else:
        eta_str = "Calculating..."

    return elapsed_str, eta_str

def inference_jhmdb(dataset_path, output_file, max_groups=None, method='sam2'):

    tokenizer, evfsam = init_models()
    # initialize Alpha-CLIP
    alphaclip_path = 'weights/alphaclip/ViT-L-14-336px.pt'
    clip, clip_preprocess = alphaclip.load(alphaclip_path,
                                           alpha_vision_ckpt_pth='weights/alphaclip/clip_l14_336_grit_20m_4xe.pth',
                                           device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])


    # checkpoint = "weights/sam2/sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    # predictor = init_propagation(method, 'jhmdb_config')
    predictor = Sam2Processor(device='cuda')

    # cutie = get_default_model(config='ytvos_config')
    # predictor = InferenceCore(cutie, cfg=cutie.cfg)
    # save_path_prefix = os.path.join(output_dir, 'JHMDB_Sentences')
    # os.makedirs(save_path_prefix, exist_ok=True)

    print("Loading JHMDB annotations...")
    annotations = load_jhmdb_annotations(dataset_path)
    groups = {}
    for video_id, frame_path, mask_path, instance_id, text_query in annotations:
        key = (video_id, int(instance_id), " ".join(str(text_query).lower().split()))
        if key not in groups:
            groups[key] = set()
        frame_name = os.path.basename(frame_path)
        frame_idx = int(frame_name.split('.')[0])
        groups[key].add(frame_idx)

    group_items = list(groups.items())
    if max_groups is not None:
        group_items = group_items[:max_groups]
    print(f"Total {len(group_items)} targets")

    predictions = []
    processed = 0
    failed = 0
    start_time = time.time()
    for idx, (key, requested_frames) in enumerate(group_items):
        try:
            video_id, instance_id, text_query = key

            video_dir = get_video_dir(video_id, dataset_path)
            if video_dir is None:
                raise FileNotFoundError(f"Video directory not found for {video_id}")
            frames_np = read_all_frames(video_id, dataset_path)
            video_len = len(frames_np)

            imgs_beit = []
            imgs_sam = []
            imgs_clip = []
            imgs_cutie = []
            original_sizes = []
            all_blur_scores=[]
            for i in range(video_len):
                image_np = frames_np[i]
                img_pil = Image.fromarray(image_np)
                original_sizes.append(image_np.shape[:2])

                img_beit = beit3_preprocess(img_pil, 224)
                imgs_beit.append(img_beit)

                img_sam, resize_shape = sam_preprocess(image_np)
                imgs_sam.append(img_sam)

                img_clip = clip_preprocess(img_pil)
                imgs_clip.append(img_clip)

                sharpness_score = get_fft_sharpness(image_np)
                all_blur_scores.append(sharpness_score)

                if method == 'cutie':
                    img_cutie = tv.transforms.ToTensor()(img_pil)
                    imgs_cutie.append(img_cutie)
            all_imgs_clip_tensor = torch.stack(imgs_clip, dim=0).cuda()
            ref_masks = []
            ref_scores = []
            ref_num = 10

            ref_masks, ref_scores, sample_indices = get_ref_frames_batch(evfsam, tokenizer, clip, clip_preprocess_mask,
                                                                         text_query,
                                                                         imgs_sam, imgs_beit, imgs_clip, resize_shape,
                                                                         original_sizes,
                                                                         video_len, ref_num, all_blur_scores,
                                                                         all_imgs_clip_tensor, sample=False)

            best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)

            # best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1))
            best_i = sample_indices[best_ref_idx]

            # mask_path = None
            # for ann_video_id, frame_path, ann_mask_path, ann_instance_id, ann_text_query in annotations:
            #     if ann_video_id == video_id and ann_instance_id == instance_id:
            #         mask_path = ann_mask_path
            #         break
            #
            # if mask_path:
            #     category = os.path.basename(os.path.dirname(os.path.dirname(mask_path)))
            # else:
            #     category = 'unknown'
            #
            # sample_dir = os.path.join(save_path_prefix, category, video_id)
            # os.makedirs(sample_dir, exist_ok=True)

            if method == 'sam2':
                video_frames = []
                for i in range(video_len):
                    video_frames.append(Image.fromarray(frames_np[i]))
                
                mask_2d = ref_masks[best_ref_idx].squeeze(0).cpu().numpy().astype(np.float32)
                obj_ids = [1]
                
                masks = predictor.propagate(
                    frames=video_frames,
                    obj_ids=obj_ids,
                    mask=mask_2d,
                    ann_idx=best_i
                )
                
                for frame_idx, mask in enumerate(masks):
                    frame_1_based = frame_idx + 1
                    if frame_1_based in requested_frames:
                        mask_np_tensor = mask.detach().cpu() if torch.is_tensor(mask) else torch.from_numpy(mask)
                        target_h, target_w = original_sizes[frame_idx]
                        # mask_up = F.interpolate(mask_np_tensor.float().unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='nearest').squeeze(0).squeeze(0)
                        mask_up = F.interpolate(
                            mask_np_tensor.float().unsqueeze(0).unsqueeze(0),
                            size=(target_h, target_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                        # Apply a threshold to get a clean binary mask
                        mask_up = (mask_up > 0.0).float()

                        mask_np = mask_up.detach().cpu().numpy().astype(np.uint8)
                        rle = mask_util.encode(np.asfortranarray(mask_np))
                        rle['counts'] = rle['counts'].decode('ascii')
                        image_id = get_image_id(video_id, frame_1_based)
                        # confidence_score = float(mask_up.mean().item())
                        confidence_score = float(ref_scores[best_ref_idx].item())
                        predictions.append({
                            'image_id': image_id, 'category_id': 1,
                            'segmentation': rle, 'score': confidence_score
                        })

            elif method == 'cutie':
                def process_cutie_mask_old(i_, mask_prob_, mask_):
                    frame_1_based = i_ + 1
                    if frame_1_based in requested_frames:
                        target_h, target_w = original_sizes[i_]
                        mask_up = F.interpolate(mask_.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='nearest').squeeze(0).squeeze(0)
                        mask_np = mask_up.detach().cpu().numpy().astype(np.uint8)
                        rle = mask_util.encode(np.asfortranarray(mask_np))
                        rle['counts'] = rle['counts'].decode('ascii')
                        image_id = get_image_id(video_id, frame_1_based)

                        fg_logits = mask_prob_[1].float()
                        fg_logits_up = F.interpolate(fg_logits.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                        fg_vals = fg_logits_up[mask_up > 0.5]
                        if fg_vals.numel() > 0:
                            mean_logit = fg_vals.mean()
                            confidence_score = float(torch.sigmoid(mean_logit).item())
                        else:
                            confidence_score = float(torch.sigmoid(fg_logits_up.mean()).item())
                        predictions.append({
                            'image_id': image_id, 'category_id': 1,
                            'segmentation': rle, 'score': confidence_score
                        })
                        # mask_img = mask.detach().cpu().numpy().astype(np.float32)
                        # mask_img = Image.fromarray(mask_img * 255).convert('L')
                        # save_file = os.path.join(sample_dir, f'{frame_1_based:05d}.png')
                        # mask_img.save(save_file)

                def process_cutie_mask(i_, mask_prob_):
                    frame_1_based = i_ + 1
                    if frame_1_based in requested_frames:
                        target_h, target_w = original_sizes[i_]

                        fg_logits = mask_prob_[1].float()

                        fg_logits_up = F.interpolate(
                            fg_logits.unsqueeze(0).unsqueeze(0),
                            size=(target_h, target_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)

                        mask_up_binary = (fg_logits_up > 0.5).float()

                        mask_np = mask_up_binary.detach().cpu().numpy().astype(np.uint8)
                        rle = mask_util.encode(np.asfortranarray(mask_np))
                        rle['counts'] = rle['counts'].decode('ascii')
                        image_id = get_image_id(video_id, frame_1_based)

                        confidence_score = float(ref_scores[best_ref_idx].item())

                        predictions.append({
                            'image_id': image_id, 'category_id': 1,
                            'segmentation': rle, 'score': confidence_score
                        })
                for i in range(best_i, video_len):
                    if i == best_i:
                        mask_prob = predictor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                    else:
                        mask_prob = predictor.step(imgs_cutie[i].cuda())
                    mask = predictor.output_prob_to_mask(mask_prob).float()
                    # process_cutie_mask(i, mask_prob, mask)
                    process_cutie_mask(i, mask_prob)
                    if i == video_len - 1:
                        predictor.clear_memory()

                for i in range(best_i, -1, -1):
                    if i == best_i:
                        mask_prob = predictor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                    else:
                        mask_prob = predictor.step(imgs_cutie[i].cuda())
                    mask = predictor.output_prob_to_mask(mask_prob).float()
                    # process_cutie_mask(i, mask_prob, mask)
                    process_cutie_mask(i, mask_prob)
                    if i == 0:
                        predictor.clear_memory()

            processed += 1

        except Exception as e:
            print(f"Error while processing target  {idx + 1} targets: {e}")
            failed += 1
            continue

        done = idx + 1
        total = len(group_items)
        elapsed_str, eta_str = format_elapsed_eta(start_time, done, total)
        print(f'Target progress {done}/{total} | elapsed: {elapsed_str} | ETA: {eta_str}')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f)

    print("\nInference complete!")
    print(f"Success: {processed}")
    print(f"Failed: {failed}")
    print(f"COCO predictions saved to: {output_file}")


def main():
    DEFAULT_DATASET_PATH = '../../dataset/jhmdb_sentences'
    DEFAULT_OUTPUT_DIR = 'outputs/Jhmdb_sam'

    parser = argparse.ArgumentParser(description='JHMDB-Sentences inference and export predictions in official COCO format')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH,
                        help=f'JHMDB-Sentencespath (default: {DEFAULT_DATASET_PATH})')
    parser.add_argument('--output_file', type=str, default=None,
                        help=f'COCO prediction output file (default: jhmdb_[method].json)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'image (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--max_groups', type=int, default=None,
                        help='maximum number of targets to process (for debugging)')
    parser.add_argument('-m', '--method', type=str, default='sam2',
                        choices=['sam2', 'cutie'],
                        help='propagation method: sam2  cutie (default: sam2)')

    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(args.output_dir, f'jhmdb_2_{args.method}_5.json')

    print("=" * 60)
    print("JHMDB-Sentences Configuration:")
    print(f"  path: {args.dataset_path}")
    print(f"  propagation method: {args.method.upper()}")
    print(f"  prediction output: {output_file}")
    print(f"  image: {args.output_dir}")
    if args.max_groups:
        print(f"  large objects: {args.max_groups}")
    else:
        print("  large objects: all")
    print("=" * 60)

    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        inference_jhmdb(args.dataset_path, output_file, args.max_groups, args.method)


if __name__ == '__main__':
    main()

