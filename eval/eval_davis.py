import os
import sys
from time import time
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from datasets.davis2017 import DAVISEvaluation


def evaluate_single_folder(folder_info):
    """
    Evaluate one result folder; used by multiprocessing.
    """
    folder_name, davis_path, base_result_path, task, gt_set = folder_info
    results_path = os.path.join(base_result_path, folder_name)

    if not os.path.exists(results_path):
        return None

    dataset_eval = DAVISEvaluation(davis_root=davis_path, task=task, gt_set=gt_set)
    metrics_res = dataset_eval.evaluate(results_path)
    J, F = metrics_res['J'], metrics_res['F']

    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    return {
        'Folder': folder_name,
        'J&F-Mean': final_mean,
        'J-Mean': np.mean(J["M"]),
        'F-Mean': np.mean(F["M"]),
        'J-Recall': np.mean(J["R"]),
        'F-Recall': np.mean(F["R"])
    }


if __name__ == '__main__':
    default_davis_path = '../../dataset/davis/DAVIS'
    base_result_path = 'outputs/davis_mamba/Annotations_origin_sam_2'
    anno_folders = ['anno_0', 'anno_1', 'anno_2', 'anno_3']

    parser = argparse.ArgumentParser()
    parser.add_argument('--davis_path', type=str, default=default_davis_path)
    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--task', type=str, default='unsupervised')
    args, _ = parser.parse_known_args()

    time_start = time()

    tasks = [(f, args.davis_path, base_result_path, args.task, args.set) for f in anno_folders]

    print(f"Starting parallel evaluation; folder count: {len(anno_folders)}")

    with Pool(processes=len(anno_folders)) as pool:
        results = pool.map(evaluate_single_folder, tasks)

    all_global_results = [r for r in results if r is not None]

    if all_global_results:
        print("\n" + "=" * 60)
        print("Final batch evaluation report (Final Batch Report)")
        print("=" * 60)

        df_final = pd.DataFrame(all_global_results)
        df_final = df_final.sort_values(by='Folder')

        numeric_cols = ['J&F-Mean', 'J-Mean', 'F-Mean', 'J-Recall', 'F-Recall']
        avg_values = df_final[numeric_cols].mean()

        avg_row = pd.DataFrame([{
            'Folder': 'OVERALL_AVERAGE',
            'J&F-Mean': avg_values['J&F-Mean'],
            'J-Mean': avg_values['J-Mean'],
            'F-Mean': avg_values['F-Mean'],
            'J-Recall': avg_values['J-Recall'],
            'F-Recall': avg_values['F-Recall']
        }])

        df_with_avg = pd.concat([df_final, avg_row], ignore_index=True)

        print(df_with_avg.to_string(index=False))

        summary_path = os.path.join(base_result_path, f'origin_sam_2.csv')
        df_with_avg.to_csv(summary_path, index=False, float_format="%.5f")

        print("-" * 40)
        print(f"Summary results with averages saved to: {summary_path}")

    total_time = time() - time_start
    print(f'\nTotal parallel evaluation time: {total_time:.2f} seconds')
