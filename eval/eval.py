import sys
import os
import io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from tqdm import tqdm
from pycocotools.mask import decode

def compute_mask_iou(outputs: torch.Tensor, labels: torch.Tensor, EPS=1e-6):
    outputs = outputs.int()
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
    iou = (intersection + EPS) / (union + EPS)  # EPS is used to avoid division by zero
    return iou, intersection, union

# mask
def calculate_precision_at_k_and_iou_metrics(coco_gt: COCO, coco_pred: COCO):
    counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_intersection_area = 0
    total_union_area = 0
    ious_list = []
    for instance in tqdm(coco_gt.imgs.keys()):  # each image_id contains exactly one instance
        gt_annot = coco_gt.imgToAnns[instance][0]
        gt_mask = decode(gt_annot['segmentation'])
        pred_annots = coco_pred.imgToAnns[instance]
        if not pred_annots:
            print(f"\nWarning: image {instance} has no prediction result; skipped.")
            continue
        pred_annot = sorted(pred_annots, key=lambda a: a['score'])[-1]  # choose pred with highest score
        pred_mask = decode(pred_annot['segmentation'])
        iou, intersection, union = compute_mask_iou(torch.tensor(pred_mask).unsqueeze(0),
                                               torch.tensor(gt_mask).unsqueeze(0))
        iou, intersection, union = iou.item(), intersection.item(), union.item()
        for iou_threshold in counters_by_iou.keys():
            if iou > iou_threshold:
                counters_by_iou[iou_threshold] += 1
        total_intersection_area += intersection
        total_union_area += union
        ious_list.append(iou)
    num_samples = len(ious_list)
    precision_at_k = np.array(list(counters_by_iou.values())) / num_samples
    overall_iou = total_intersection_area / total_union_area
    mean_iou = np.mean(ious_list)
    return precision_at_k, overall_iou, mean_iou

def evaluate_dataset(gt_path, pred_path, dataset_name):
    """
    Args:
        gt_path: COCOground truthpath
        pred_path: COCOResultpath
        dataset_name: dataset name ('a2d'  'jhmdb')
    Returns:
        dict: evaluation metric dictionary
    """
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()

    coco_gt = COCO(gt_path)
    
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []
    
    coco_pred = coco_gt.loadRes(pred_path)
    
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
    
    coco_eval.params.useCats = 0
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    sys.stdout = stdout_backup

    ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
    ap_metrics = coco_eval.stats[:6]
    eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
    
    precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
    eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
    eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
    
    return eval_metrics


def print_results(eval_metrics, dataset_name):
    """
    Args:
        eval_metrics: evaluation metric dictionary
        dataset_name: dataset name
    """
    print(f"\n{'='*60}")
    print(f'{dataset_name.upper()} dataset evaluation results')
    print(f"{'='*60}")
    
    print('\nAverage Precision (Average Precision) metrics:')
    print(f"  mAP @ IoU=0.50:0.95: {eval_metrics['mAP 0.5:0.95']:.4f}")
    print(f"  AP @ IoU=0.50:      {eval_metrics['AP 0.5']:.4f}")
    print(f"  AP @ IoU=0.75:      {eval_metrics['AP 0.75']:.4f}")
    print(f"  AP @ IoU=0.50:0.95 (small objects): {eval_metrics['AP 0.5:0.95 S']:.4f}")
    print(f"  AP @ IoU=0.50:0.95 (medium objects): {eval_metrics['AP 0.5:0.95 M']:.4f}")
    print(f"  AP @ IoU=0.50:0.95 (large objects): {eval_metrics['AP 0.5:0.95 L']:.4f}")
    
    print('\nPrecision (Precision) metrics:')
    print(f"  P@0.5: {eval_metrics['P@0.5']:.4f}")
    print(f"  P@0.6: {eval_metrics['P@0.6']:.4f}")
    print(f"  P@0.7: {eval_metrics['P@0.7']:.4f}")
    print(f"  P@0.8: {eval_metrics['P@0.8']:.4f}")
    print(f"  P@0.9: {eval_metrics['P@0.9']:.4f}")
    
    print('\nIoU (Intersection over Union) metrics:')
    print(f"  Mean IoU (Mean IoU):     {eval_metrics['mean_iou']:.4f}")
    print(f"  Overall IoU (Overall IoU):  {eval_metrics['overall_iou']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='A2D and JHMDB dataset evaluation')
    
    parser.add_argument('--a2d_gt', type=str, 
                        default='../../dataset/a2d_sentences/a2d_sentences_test_annotations_in_coco_format.json',
                        help='A2DCOCOGTpath')
    parser.add_argument('--a2d_pred', type=str, default='outputs/A2d/a2d_2_sam2_5.json',
                        help='A2DCOCOpath')
    
    parser.add_argument('--jhmdb_gt', type=str,
                        default='../../dataset/jhmdb_sentences/jhmdb_sentences_gt_annotations_in_coco_format.json',
                        help='JHMDBCOCOGTpath')
    parser.add_argument('--jhmdb_pred', type=str, default='outputs/Jhmdb_sam/jhmdb_2_sam2_5.json',
                        help='JHMDBCOCOpath')
    
    parser.add_argument('--dataset', type=str, choices=['a2d', 'jhmdb', 'both'], default='both',
                        help=': a2d, jhmdb,  both')
    
    args = parser.parse_args()
    
    if args.dataset in ['a2d', 'both']:
        if not os.path.exists(args.a2d_gt):
            print(f"Error: A2D GTFile not found: {args.a2d_gt}")
            return
        if not os.path.exists(args.a2d_pred):
            print(f"Error: A2D File not found: {args.a2d_pred}")
            return
    
    if args.dataset in ['jhmdb', 'both']:
        if not os.path.exists(args.jhmdb_gt):
            print(f"Error: JHMDB GTFile not found: {args.jhmdb_gt}")
            return
        if not os.path.exists(args.jhmdb_pred):
            print(f"Error: JHMDB File not found: {args.jhmdb_pred}")
            return
    
    if args.dataset in ['a2d', 'both']:
        a2d_metrics = evaluate_dataset(args.a2d_gt, args.a2d_pred, 'a2d')
        print_results(a2d_metrics, 'a2d')
    
    if args.dataset in ['jhmdb', 'both']:
        jhmdb_metrics = evaluate_dataset(args.jhmdb_gt, args.jhmdb_pred, 'jhmdb')
        print_results(jhmdb_metrics, 'jhmdb')


if __name__ == '__main__':
    main()
