import pandas as pd
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    results_tot = {}
    overall_results = []
    for method in ['ablation_leiden_upgma_sii', 'ablation_leiden_nj_sii', 'ablation_leiden_mst_dbi', 'ablation_leiden_mst_chi']:
        overall_results = []
        folder = os.path.join('results/rebuttal/20251020/data_large2', method, 'CNV', 'leiden')
        datafolder = os.path.join('data', 'data_large2')
        levels = [0.25, 0.3, 0.35, 0.4]
        num_files = len(os.listdir(folder))
        for noise_level in levels:
            merged_curr_level = []
            for dir in os.listdir(folder):

                if dir.endswith(str(noise_level)):
                    merged_curr_level.append(pd.read_csv(os.path.join(folder, dir, f'result.csv'), index_col=None))
                

            if len(merged_curr_level) > 0:
                merged_curr_level = pd.concat(merged_curr_level)
                result_metrics = merged_curr_level[['sc1b', 'sc2', 'sc3']]

                overall_results.append(result_metrics)
        
        overall_results = np.concatenate(overall_results)
        print(method, overall_results.mean(0))