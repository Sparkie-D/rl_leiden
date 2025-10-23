import os
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ubuntu/duxinghao/clone/rl_leiden')
from src.utils import load_data
from src.evaluation.evaluate import evaluate

if __name__ == '__main__':
    data_path = '/home/ubuntu/duxinghao/clone/data/data_large2'
    result_path = '/home/ubuntu/duxinghao/clone/rl_leiden/results/old_logs/data_large2/rl_leiden/CNV/leiden'
    result_names = os.listdir(result_path)
    
    for dataset_id in range(0, 500):
        data, c2cl, dataset_name, gt_tree, cl2idx  = load_data(data_path, dataset_id)
        if dataset_name not in result_names:
            continue
        print(f"processing dataset {dataset_name}")
        cl2idx = {x:i for i, x in enumerate(set(c2cl.clone))}
        true_labels = c2cl.clone.map(cl2idx).values
        pred_labels = pd.read_csv(os.path.join(result_path, dataset_name, 'cell2cluster.csv'))['cluster'].values

        results = evaluate(cnv_data=data.values, 
                           true_labels=true_labels, 
                           pred_labels=pred_labels, 
                           tree_method='mst',
                           real_tree=gt_tree)

        df = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
        df.to_csv(os.path.join(result_path, dataset_name, 'result_new.csv'))