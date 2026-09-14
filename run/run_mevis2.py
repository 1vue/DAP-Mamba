import time

import alphaclip
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
import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from tqdm import tqdm
warnings.filterwarnings('ignore')
import alphaclip
from alphaclip import alpha_clip
from models.sam2.sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt

PROFILE_MODULES = [
    "preprocessing",
    "dhas_fft",
    "dhas_alpha_clip_image",
    "dhas_text_scoring",
    "reference_segmentation",
    "sam2_state_init",
    "vos_propagation",
    "postprocessing",
]

REPORT_MODULES = {
    "pre_post_processing": ["preprocessing", "postprocessing"],
    "dhas_quality_scoring": ["dhas_fft"],
    "dhas_semantic_scoring": ["dhas_alpha_clip_image", "dhas_text_scoring"],
    "reference_segmentation": ["reference_segmentation"],
    "sam2_video_propagation": ["sam2_state_init", "vos_propagation"],
}


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_section(bucket, name):
    cuda_sync()
    start = time.perf_counter()
    yield
    cuda_sync()
    bucket[name] += time.perf_counter() - start
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

def save_prediction(frame_idx, logits, save_dir, frame_names):
    pred_mask = (logits[0] > 0.0).cpu().numpy().astype(np.float32).squeeze()

    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert('L')

    save_file = os.path.join(save_dir, frame_names[frame_idx] + '.png')
    mask_img.save(save_file)


def save_or_convert_prediction(frame_idx, logits, save_dir, frame_names, save_masks):
    pred_mask = (logits[0] > 0.0).cpu().numpy().astype(np.float32).squeeze()
    if save_masks:
        os.makedirs(save_dir, exist_ok=True)
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert('L')
        save_file = os.path.join(save_dir, frame_names[frame_idx] + '.png')
        mask_img.save(save_file)
    return pred_mask


def uniform_sample_indices(video_len, ref_num):
    if video_len <= 0:
        return []
    if ref_num <= 1:
        return [0]
    if video_len <= ref_num:
        return list(range(video_len))
    return [int(i * (video_len - 1) / (ref_num - 1)) for i in range(ref_num)]


def min_max_normalize_tensor(tensor):
    if tensor.numel() == 0:
        return tensor
    if torch.isclose(tensor.max(), tensor.min()):
        return torch.ones_like(tensor) * 0.5
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)


def dhas_sample_indices_from_features(all_blur_scores, all_image_features, text_features, ref_num):
    video_len = len(all_blur_scores)
    sample_indices = []
    chunks = np.array_split(np.arange(video_len), min(ref_num, video_len))

    for chunk in chunks:
        chunk_indices = chunk.tolist()
        chunk_clip_feats = all_image_features[chunk_indices]
        clip_scores = (chunk_clip_feats @ text_features.T).squeeze(-1)
        fft_scores = torch.tensor(
            [all_blur_scores[i] for i in chunk_indices],
            device=clip_scores.device,
            dtype=clip_scores.dtype,
        )
        norm_clip = min_max_normalize_tensor(clip_scores)
        norm_fft = min_max_normalize_tensor(fft_scores)
        combined_scores = 0.7 * norm_clip + 0.3 * norm_fft
        best_local_idx = torch.argmax(combined_scores).item()
        sample_indices.append(chunk_indices[best_local_idx])

    return sample_indices


def load_mevis_profile_metadata(root, max_videos):
    meta_file = os.path.join(root, 'valid', 'meta_expressions.json')
    with open(meta_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['videos']

    video_list = sorted(data.keys())
    if max_videos > 0:
        video_list = video_list[:max_videos]
    return data, video_list


def prepare_video_inputs_for_profile(img_folder, video_name, frames, strategy, clip_preprocess):
    timings = defaultdict(float)
    imgs_beit, imgs_sam, imgs_clip = [], [], []
    all_blur_scores = []
    resize_shape = None
    original_size_list = None

    for frame_name in frames:
        img_path = os.path.join(img_folder, video_name, frame_name + '.jpg')
        with timed_section(timings, "preprocessing"):
            pil_img = Image.open(img_path).convert('RGB')
            image_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            imgs_beit.append(beit3_preprocess(pil_img, 224))
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)
            imgs_clip.append(clip_preprocess(pil_img))
            original_size_list = [image_np.shape[:2]]

        if strategy == "dhas":
            with timed_section(timings, "dhas_fft"):
                all_blur_scores.append(get_fft_sharpness(image_np))

    return {
        "timings": timings,
        "imgs_beit": imgs_beit,
        "imgs_sam": imgs_sam,
        "imgs_clip": imgs_clip,
        "all_blur_scores": all_blur_scores,
        "resize_shape": resize_shape,
        "original_size_list": original_size_list,
    }


def encode_dhas_video_features(imgs_clip, clip, timings):
    with timed_section(timings, "dhas_alpha_clip_image"):
        all_imgs_clip_tensor = torch.stack(imgs_clip, dim=0).cuda()
        n, _, h, w = all_imgs_clip_tensor.shape
        dummy_alpha = torch.ones(n, 1, h, w, device=all_imgs_clip_tensor.device)
        with torch.no_grad():
            all_image_features = clip.visual(all_imgs_clip_tensor, dummy_alpha)
            all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
    return all_image_features


def select_profile_indices(strategy, exp, video_len, ref_num, clip, all_blur_scores,
                           all_image_features, timings):
    if strategy == "uniform":
        return uniform_sample_indices(video_len, ref_num)

    with timed_section(timings, "dhas_text_scoring"):
        clip_text = alpha_clip.tokenize([exp]).cuda()
        with torch.no_grad():
            text_features = clip.encode_text(clip_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return dhas_sample_indices_from_features(
                all_blur_scores, all_image_features, text_features, ref_num
            )


def generate_profile_reference_masks(evfsam, tokenizer, clip, clip_preprocess_mask, exp,
                                     sample_indices, imgs_sam, imgs_beit, imgs_clip,
                                     resize_shape, original_size_list, timings):
    with timed_section(timings, "reference_segmentation"):
        batch_imgs_sam = torch.stack([imgs_sam[i] for i in sample_indices]).cuda()
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
                num_frames=len(sample_indices),
                sampled_indices=torch.tensor(sample_indices)
            )

            clip_text = alpha_clip.tokenize([exp]).cuda()
            text_features = clip.encode_text(clip_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            ref_masks = []
            ref_scores = []
            for idx, frame_idx in enumerate(sample_indices):
                ref_mask = (batch_ref_masks[idx] > 0).float()
                ref_masks.append(ref_mask)

                alpha = clip_preprocess_mask(ref_mask).cuda()
                image_features = clip.visual(imgs_clip[frame_idx].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vt_alignment = torch.matmul(image_features, text_features.transpose(0, 1))[0]
                ref_score = 0.5 * batch_iou_scores[idx] + 0.5 * vt_alignment
                ref_scores.append(ref_score.reshape(-1)[0])

    return ref_masks, ref_scores


def profile_sam2_propagation(predictor, inference_state, ref_masks, ref_scores, sample_indices,
                             frames, save_path, save_masks, timings):
    best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0)).item()
    best_i = sample_indices[best_ref_idx]
    best_mask = ref_masks[best_ref_idx].squeeze(0)

    with timed_section(timings, "vos_propagation"):
        predictor.reset_state(inference_state)
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=best_i,
            obj_id=1,
            mask=best_mask.cuda()
        )

    forward_iter = predictor.propagate_in_video(inference_state, start_frame_idx=best_i)
    while True:
        with timed_section(timings, "vos_propagation"):
            try:
                out_frame_idx, out_obj_ids, out_mask_logits = next(forward_iter)
            except StopIteration:
                break
        with timed_section(timings, "postprocessing"):
            save_or_convert_prediction(out_frame_idx, out_mask_logits, save_path, frames, save_masks)

    if best_i > 0:
        backward_iter = predictor.propagate_in_video(
            inference_state,
            start_frame_idx=best_i,
            reverse=True
        )
        while True:
            with timed_section(timings, "vos_propagation"):
                try:
                    out_frame_idx, out_obj_ids, out_mask_logits = next(backward_iter)
                except StopIteration:
                    break
            if out_frame_idx != best_i:
                with timed_section(timings, "postprocessing"):
                    save_or_convert_prediction(out_frame_idx, out_mask_logits, save_path, frames, save_masks)

    return best_i


def peak_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    cuda_sync()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def write_profile_outputs(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, "per_expression_runtime.csv")
    coarse_detail_path = os.path.join(output_dir, "per_expression_runtime_coarse.csv")
    summary_path = os.path.join(output_dir, "runtime_breakdown_summary.csv")
    strategy_path = os.path.join(output_dir, "strategy_summary.csv")

    fields = [
        "strategy", "video", "exp_id", "num_frames", "selected_indices",
        "best_ref_index", "peak_memory_gb", "total_runtime_s",
    ] + [f"{module}_s" for module in PROFILE_MODULES]

    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, 0.0) for field in fields}
            row["selected_indices"] = json.dumps(record["selected_indices"])
            writer.writerow(row)

    coarse_fields = [
        "strategy", "video", "exp_id", "num_frames", "selected_indices",
        "best_ref_index", "peak_memory_gb", "total_runtime_s",
    ] + [f"{module}_s" for module in REPORT_MODULES]
    with open(coarse_detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=coarse_fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, 0.0) for field in coarse_fields}
            row["selected_indices"] = json.dumps(record["selected_indices"])
            for module, raw_modules in REPORT_MODULES.items():
                row[f"{module}_s"] = sum(record[f"{raw_module}_s"] for raw_module in raw_modules)
            writer.writerow(row)

    grouped = defaultdict(list)
    for record in records:
        grouped[record["strategy"]].append(record)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "module", "total_time_s", "avg_time_per_expression_s", "percentage"],
        )
        writer.writeheader()
        for strategy, strategy_records in grouped.items():
            total_by_module = {
                module: sum(
                    sum(r[f"{raw_module}_s"] for raw_module in raw_modules)
                    for r in strategy_records
                )
                for module, raw_modules in REPORT_MODULES.items()
            }
            grand_total = sum(total_by_module.values())
            for module in REPORT_MODULES:
                total_time = total_by_module[module]
                writer.writerow({
                    "strategy": strategy,
                    "module": module,
                    "total_time_s": total_time,
                    "avg_time_per_expression_s": total_time / max(len(strategy_records), 1),
                    "percentage": 100.0 * total_time / grand_total if grand_total > 0 else 0.0,
                })

    with open(strategy_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy", "num_expressions", "total_runtime_s",
                "avg_runtime_per_expression_s", "peak_memory_gb",
            ],
        )
        writer.writeheader()
        for strategy, strategy_records in grouped.items():
            total_runtime = sum(r["total_runtime_s"] for r in strategy_records)
            writer.writerow({
                "strategy": strategy,
                "num_expressions": len(strategy_records),
                "total_runtime_s": total_runtime,
                "avg_runtime_per_expression_s": total_runtime / max(len(strategy_records), 1),
                "peak_memory_gb": max(r["peak_memory_gb"] for r in strategy_records),
            })

    plot_profile_breakdown(grouped, output_dir)
    return detail_path, coarse_detail_path, summary_path, strategy_path


def plot_profile_breakdown(grouped, output_dir):
    colors = {
        "pre_post_processing": "#4E79A7",
        "dhas_quality_scoring": "#F28E2B",
        "dhas_semantic_scoring": "#E15759",
        "reference_segmentation": "#59A14F",
        "sam2_video_propagation": "#EDC948",
    }
    strategies = list(grouped.keys())
    bottoms = np.zeros(len(strategies))

    plt.figure(figsize=(9.5, 4.8))
    for module, raw_modules in REPORT_MODULES.items():
        values = []
        for strategy in strategies:
            records = grouped[strategy]
            values.append(
                sum(
                    sum(r[f"{raw_module}_s"] for raw_module in raw_modules)
                    for r in records
                ) / max(len(records), 1)
            )
        plt.bar(strategies, values, bottom=bottoms, label=module, color=colors[module])
        bottoms += np.array(values)
    plt.ylabel("Average runtime per expression (s)")
    plt.xlabel("Frame selection strategy")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_breakdown_stacked_bar.png"), dpi=400)
    plt.close()

    if "dhas" in grouped:
        records = grouped["dhas"]
        values = [
            sum(
                sum(r[f"{raw_module}_s"] for raw_module in raw_modules)
                for r in records
            )
            for raw_modules in REPORT_MODULES.values()
        ]
        nonzero = [(module, value) for module, value in zip(REPORT_MODULES.keys(), values) if value > 0]
        if nonzero:
            labels, pie_values = zip(*nonzero)
            plt.figure(figsize=(6.2, 5.2))
            plt.pie(
                pie_values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=[colors[label] for label in labels],
            )
            plt.title("DHAS + SAM2 inference-time breakdown on MeViS")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "runtime_breakdown_dhas_pie.png"), dpi=400)
            plt.close()


def profile_efficiency(args):
    tokenizer, evfsam = init_models()
    clip, clip_preprocess = alphaclip.load(
        args.alphaclip_model,
        alpha_vision_ckpt_pth=args.alpha_vision_ckpt,
        device='cuda'
    )
    clip_preprocess_mask = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.Normalize(0.5, 0.26)
    ])

    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_checkpoint)
    data, video_list = load_mevis_profile_metadata(args.root, args.max_videos)
    img_folder = os.path.join(args.root, 'valid', 'JPEGImages')
    strategies = ["uniform", "dhas"] if args.strategy == "both" else [args.strategy]

    all_records = []
    for strategy in strategies:
        progress_bar = tqdm(video_list, desc=f"Profiling MeViS {strategy}", unit="video", file=sys.stdout)
        for video in progress_bar:
            expressions = data[video]['expressions']
            expression_list = list(expressions.keys())
            if args.max_expressions > 0:
                expression_list = expression_list[:args.max_expressions]

            frames = data[video]['frames']
            video_len = len(frames)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            prepared = prepare_video_inputs_for_profile(
                img_folder, video, frames, strategy, clip_preprocess
            )
            with timed_section(prepared["timings"], "sam2_state_init"):
                inference_state = predictor.init_state(video_path=os.path.join(img_folder, video))

            all_image_features = None
            if strategy == "dhas":
                all_image_features = encode_dhas_video_features(
                    prepared["imgs_clip"], clip, prepared["timings"]
                )
            video_level_peak_gb = peak_memory_gb()
            common_time_share = max(len(expression_list), 1)

            for exp_id in expression_list:
                exp = expressions[exp_id]['exp']
                timings = defaultdict(float)
                for module, value in prepared["timings"].items():
                    timings[module] += value / common_time_share

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                sample_indices = select_profile_indices(
                    strategy,
                    exp,
                    video_len,
                    args.ref_num,
                    clip,
                    prepared["all_blur_scores"],
                    all_image_features,
                    timings,
                )
                ref_masks, ref_scores = generate_profile_reference_masks(
                    evfsam,
                    tokenizer,
                    clip,
                    clip_preprocess_mask,
                    exp,
                    sample_indices,
                    prepared["imgs_sam"],
                    prepared["imgs_beit"],
                    prepared["imgs_clip"],
                    prepared["resize_shape"],
                    prepared["original_size_list"],
                    timings,
                )

                save_path = os.path.join(args.output_dir, "masks", strategy, video, exp_id)
                best_i = profile_sam2_propagation(
                    predictor,
                    inference_state,
                    ref_masks,
                    ref_scores,
                    sample_indices,
                    frames,
                    save_path,
                    args.save_masks,
                    timings,
                )
                expression_peak_gb = peak_memory_gb()

                record = {
                    "strategy": strategy,
                    "video": video,
                    "exp_id": exp_id,
                    "num_frames": video_len,
                    "selected_indices": sample_indices,
                    "best_ref_index": best_i,
                    "peak_memory_gb": max(video_level_peak_gb, expression_peak_gb),
                }
                for module in PROFILE_MODULES:
                    record[f"{module}_s"] = timings[module]
                record["total_runtime_s"] = sum(record[f"{module}_s"] for module in PROFILE_MODULES)
                all_records.append(record)

            del prepared
            if all_image_features is not None:
                del all_image_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    detail_path, coarse_detail_path, summary_path, strategy_path = write_profile_outputs(all_records, args.output_dir)
    print(f"[Done] Per-expression runtime: {detail_path}")
    print(f"[Done] Coarse per-expression runtime: {coarse_detail_path}")
    print(f"[Done] Module breakdown: {summary_path}")
    print(f"[Done] Strategy summary: {strategy_path}")


def test():

    # initialize DAP-Mamba
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP
    alphaclip_path = 'weights/alphaclip/ViT-L-14-336px.pt'
    clip, clip_preprocess = alphaclip.load(alphaclip_path, alpha_vision_ckpt_pth='weights/alphaclip/clip_l14_336_grit_20m_4xe.pth', device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    checkpoint = "weights/sam2/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    print("SAM2 initialized successfully")

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir + '/experiment', 'fps_test')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = '../../dataset/mevis'
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    meta_file = os.path.join(root, 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_videos = set(data.keys())
    video_list = sorted([video for video in valid_videos])

    # inference
    start_index = 3
    progress_bar = tqdm(enumerate(video_list[start_index:start_index+1]),
                        total=len(video_list[start_index:start_index+1]),
                        desc='Inference on videos',
                        unit='video', file=sys.stdout)
    total_frames = 0
    total_time_end_to_end = 0
    total_time_propagation = 0
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
        all_blur_scores = []
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
        current_video_path = os.path.join(img_folder, video)
        inference_state = predictor.init_state(video_path=current_video_path)
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

            torch.cuda.synchronize()
            inference_start_time = time.time()
            # per-frame mask prediction

            ref_num = 10
            ref_masks, ref_scores, sample_indices = get_ref_frames_batch(evfsam, tokenizer, clip, clip_preprocess_mask,
                                                                         exp,
                                                                         imgs_sam, imgs_beit, imgs_clip, resize_shape,
                                                                         original_size_list,
                                                                         video_len, ref_num, all_blur_scores,
                                                                         all_imgs_clip_tensor, video_name, exp_id,
                                                                         save_path, sample=False)

            # select reference frame with highest mask score
            best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)
            # best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1))
            best_i = sample_indices[best_ref_idx]
            predictor.reset_state(inference_state)
            best_mask = ref_masks[best_ref_idx].squeeze(0)
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=best_i,
                obj_id=1,
                mask=best_mask.cuda()
            )

            torch.cuda.synchronize()
            prop_start_time = time.time()

            # forward pass
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=best_i
            ):
                save_prediction(out_frame_idx, out_mask_logits, save_path, frames)

            # backward pass
            if best_i > 0:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=best_i,
                        reverse=True
                ):
                    if out_frame_idx != best_i:
                        save_prediction(out_frame_idx, out_mask_logits, save_path, frames)
            torch.cuda.synchronize()
            prop_end_time = time.time()
            inference_end_time = time.time()
            total_frames += video_len
            total_time_propagation += (prop_end_time - prop_start_time)
            total_time_end_to_end += (inference_end_time - inference_start_time)
    print(f"\n" + "=" * 30)
    print(f"FPS report (total frames: {total_frames})")
    print(f"Propagation Only FPS: {total_frames / total_time_propagation:.2f}")
    print(f"Total Inference FPS: {total_frames / total_time_end_to_end:.2f}")
    print("=" * 30)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MeViS inference or profile DHAS efficiency with SAM2.")
    parser.add_argument("--profile-efficiency", action="store_true")
    parser.add_argument("--strategy", choices=["uniform", "dhas", "both"], default="both")
    parser.add_argument("--root", default="../../dataset/mevis")
    parser.add_argument("--output-dir", default="outputs/efficiency_dhas_sam2_mevis")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-expressions", type=int, default=0)
    parser.add_argument("--ref-num", type=int, default=10)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--alphaclip-model", default="weights/alphaclip/ViT-L-14-336px.pt")
    parser.add_argument("--alpha-vision-ckpt", default="weights/alphaclip/clip_l14_336_grit_20m_4xe.pth")
    parser.add_argument("--sam2-checkpoint", default="weights/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        if args.profile_efficiency:
            profile_efficiency(args)
        else:
            test()
