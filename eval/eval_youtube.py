import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from datasets.davis2017.metrics import db_eval_iou, db_eval_boundary


def read_mask(path):
    """ Read an image and convert it to a binary 0/1 mask. """
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('L')
    mask = np.array(img)
    mask = (mask > 128).astype(np.uint8)
    return mask

# ==========================================
# ==========================================

class RefYoutubeVOSEvaluator:
    def __init__(self, meta_file, gt_root, pred_root):
        """
        meta_file: meta_expressions.json path (used to read the video and frame lists)
        gt_root: GT root (structure: video/exp_id/frame.png)
        pred_root: prediction result root (structure: video/exp_id/frame.png)
        """
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.meta_file = meta_file

        if not os.path.exists(self.meta_file):
            raise FileNotFoundError(f"❌ Meta file not found: {self.meta_file}")

        print(f"📖 Loading meta data: {self.meta_file}")
        with open(self.meta_file, 'r') as f:
            self.meta_data = json.load(f)

    def evaluate_video(self, video_name):
        """
        Evaluate all expressions for one video.
        """
        if video_name not in self.meta_data['videos']:
            return []

        video_data = self.meta_data['videos'][video_name]
        expressions = video_data['expressions']
        frames = video_data['frames']

        results_list = []

        for exp_id, exp_info in expressions.items():

            gt_exp_folder = os.path.join(self.gt_root, video_name, exp_id)
            pred_exp_folder = os.path.join(self.pred_root, video_name, exp_id)

            if not os.path.exists(gt_exp_folder):
                # print(f"Skip missing GT folder: {gt_exp_folder}")
                continue

            exp_j_scores = []
            exp_f_scores = []
            info_printed = False
            for frame_id in frames:
                frame_name = frame_id + '.png'

                gt_path = os.path.join(gt_exp_folder, frame_name)
                gt_mask = read_mask(gt_path)

                if gt_mask is None:
                    continue

                pred_path = os.path.join(pred_exp_folder, frame_name)
                pred_mask = read_mask(pred_path)

                if pred_mask is None:
                    pred_mask = np.zeros_like(gt_mask)

                if pred_mask.shape != gt_mask.shape:
                    h, w = gt_mask.shape
                    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                if not info_printed:
                    text_content = exp_info['exp'][:30] + "..." if len(exp_info['exp']) > 30 else exp_info['exp']

                    status_msg = (
                        f"📺 [Video: {video_name}] Exp: {exp_id} | \"{text_content}\"\n"
                        f"   📏 GT Shape: {gt_mask.shape} | Pred Shape: {pred_mask.shape}"
                    )
                    tqdm.write(status_msg)
                    info_printed = True

                j_score = db_eval_iou(gt_mask, pred_mask)

                if j_score == 0:
                    pred_sum = np.sum(pred_mask)
                    if np.sum(gt_mask) > 0:
                        if pred_sum > 0:
                            print(f"⚠️ [False positive] video: {video_name} | frame: {frame_id} | text: '{exp_info['exp']}'")
                            print(f"    reason: Pred has pixels but no overlap with GT; it may segment another object.")
                        else:
                            print(f"❌ [False negative] video: {video_name} | frame: {frame_id} | text: '{exp_info['exp']}'")
                            print(f"    reason: Pred is empty; the model did not identify the target.")
                    print("-" * 30)

                f_score = db_eval_boundary(gt_mask, pred_mask)

                exp_j_scores.append(j_score)
                exp_f_scores.append(f_score)

            if exp_j_scores:
                mean_j = np.mean(exp_j_scores)
                mean_f = np.mean(exp_f_scores)
                mean_jf = (mean_j + mean_f) / 2.0
                score_msg = (f"   🏆 Score -> J&F: {mean_jf:.4f}  (J: {mean_j:.4f} | F: {mean_f:.4f})")
                tqdm.write(score_msg)
                tqdm.write("-" * 50)
                results_list.append({
                    'video': video_name,
                    'exp_id': exp_id,
                    'exp_text': exp_info['exp'],
                    'J': mean_j,
                    'F': mean_f,
                    'J&F': (mean_j + mean_f) / 2.0
                })


        return results_list


# ==========================================
# ==========================================

if __name__ == '__main__':

    DATASET_ROOT = '../../dataset/ref-youtube'

    META_FILE = os.path.join(DATASET_ROOT, 'meta_expressions', 'valid', 'meta_expressions.json')

    GT_RESULT_PATH = os.path.join(DATASET_ROOT, 'valid', 'Annotations')

    # PRED_RESULT_PATH = 'outputs/full_video_new_mamba/Annotations_origin_sam_2'
    PRED_RESULT_PATH = 'other/DAP-Mamba/Ref_YTVOS_val'

    # ------------------------------------------------

    if not os.path.exists(GT_RESULT_PATH):
        print(f"❌ Error: GT path does not exist: {GT_RESULT_PATH}")
        print("Ensure GT_RESULT_PATH points to the split mask root folder")
        sys.exit(1)

    if not os.path.exists(PRED_RESULT_PATH):
        print(f"❌ Error: prediction path does not exist: {PRED_RESULT_PATH}")
        sys.exit(1)

    print("🚀 Starting Ref-Youtube-VOS evaluation (Mode: Split GT Folders)")
    print(f"📂 GT path:   {GT_RESULT_PATH}")
    print(f"📂 Pred path: {PRED_RESULT_PATH}")

    evaluator = RefYoutubeVOSEvaluator(meta_file=META_FILE, gt_root=GT_RESULT_PATH, pred_root=PRED_RESULT_PATH)

    pred_videos = sorted(os.listdir(PRED_RESULT_PATH))
    pred_videos = [v for v in pred_videos if os.path.isdir(os.path.join(PRED_RESULT_PATH, v))]

    all_results = []

    pbar = tqdm(pred_videos, desc="Evaluating Videos")
    for video_name in pbar:
        vid_results = evaluator.evaluate_video(video_name)
        if vid_results:
            all_results.extend(vid_results)

    if all_results:
        print("\n" + "=" * 60)
        print("📊 Evaluation report")
        print("=" * 60)

        df = pd.DataFrame(all_results)

        global_j = df['J'].mean()
        global_f = df['F'].mean()
        global_jf = df['J&F'].mean()

        print(f"✅ Expressions: {len(df)}")
        print(f"🏆 Global J&F : {global_jf:.5f}")
        print(f"   - Global J : {global_j:.5f}")
        print(f"   - Global F : {global_f:.5f}")

        save_csv_path = os.path.join(os.path.dirname(PRED_RESULT_PATH), 'Ref_YTVOS_val.csv')

        avg_row = pd.DataFrame([{
            'video': 'OVERALL_MEAN', 'exp_id': '-', 'exp_text': '-',
            'J': global_j, 'F': global_f, 'J&F': global_jf
        }])

        df_final = pd.concat([df, avg_row], ignore_index=True)
        df_final.to_csv(save_csv_path, index=False, float_format="%.5f")
        print(f"\n💾 Results saved: {save_csv_path}")
    else:
        print("❌ No results were generated. Check that GT and Pred share the same folder structure (video/exp_id/frame.png).")
