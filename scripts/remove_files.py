import os

def replace_files(file_a, file_b):
    """删除文件A，将文件B改名为A"""
    try:
        # 如果文件A存在则删除
        if os.path.exists(file_a):
            os.remove(file_a)
        
        # 重命名文件B为A
        os.rename(file_b, file_a)
        print(f"操作成功: {file_b} -> {file_a}")
        
    except FileNotFoundError:
        print(f"错误: {file_b} 文件不存在")
    except Exception as e:
        print(f"操作失败: {e}")

import os
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ubuntu/duxinghao/clone/rl_leiden')
from src.utils import load_data
from src.evaluation.evaluate import evaluate

if __name__ == '__main__':
    data_path = '/home/ubuntu/duxinghao/clone/data/data_large2'
    result_path = '/home/ubuntu/duxinghao/clone/rl_leiden/results/rebuttal/20251020/data_large2/ablation_leiden_upgma_sii/CNV/leiden'
    result_names = os.listdir(result_path)
    
    for dataset_id in range(0, 500):
        data, c2cl, dataset_name, gt_tree, cl2idx  = load_data(data_path, dataset_id)
        if dataset_name not in result_names:
            continue
        print(f"processing dataset {dataset_name}")
        # cl2idx = {x:i for i, x in enumerate(set(c2cl.clone))}
        # true_labels = c2cl.clone.map(cl2idx).values
        # pred_labels = pd.read_csv(os.path.join(result_path, dataset_name, 'cell2cluster.csv'))['cluster'].values

        # results = evaluate(cnv_data=data.values, 
        #                    true_labels=true_labels, 
        #                    pred_labels=pred_labels, 
        #                    tree_method='mst',
        #                    real_tree=gt_tree)

        # df = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
        # df.to_csv(os.path.join(result_path, dataset_name, 'result_new.csv'))
        src = os.path.join(result_path, dataset_name, 'result.csv')
        dst = os.path.join(result_path, dataset_name, 'result_new.csv')
        replace_files(src, dst)