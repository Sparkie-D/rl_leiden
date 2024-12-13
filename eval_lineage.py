import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import Manager
import argparse


# 函数：判断子细胞是否是父细胞的子节点
def is_subnode(parent, child):
    parent = np.array(parent)
    child = np.array(child)
    is_parent_missing = (parent == -1)
    is_child_missing = (child == -1)
    
    inherit = (child == parent) | (is_parent_missing & is_child_missing) | (parent == 0)
    if not np.all(inherit):
        return False

    changes = (child != parent) & (parent == 0)
    return np.any(changes)

# 优化的 calc_gt_pairs 函数
def calc_gt_pairs(data):
    gt_pairs = {i: [] for i in data.index}

    # 使用 Manager 来共享 is_done 数据
    with Manager() as manager:
        
        # 为并行化准备数据
        def process_pair(m, n):
            if is_subnode(data.iloc[m], data.iloc[n]):
                cell_i = data.index[m]
                cell_j = data.index[n]
                gt_pairs[cell_i].append(cell_j)

        # 使用并行化加速计算
        Parallel(n_jobs=-1)(delayed(process_pair)(m, n) for m in range(len(data.index)) for n in range(len(data.index)))
    
    return gt_pairs

# def calc_gt_pairs(data):
#     gt_pairs = {i:[] for i in data.index}
#     for m in range(len(data.index)):
#         cell_i = data.index[m]
#         for n in range(len(data.index)):
#             cell_j = data.index[n]
#             if is_subnode(data.loc[cell_i], data.loc[cell_j]):
#                 gt_pairs[cell_i].append(cell_j)
#     return gt_pairs

def save_gt_pairs_to_csv(gt_pairs, filename='gt_pairs.csv'):
    # 创建一个字典来存储每个细胞的子细胞列表
    data_to_save = []
    for i, children in gt_pairs.items():
        if len(children) > 0:
            # 将每个细胞和其子细胞列表作为一行数据保存
            data_to_save.append({'cell': i, 'children': children})

    # 将数据保存为 CSV 文件
    df = pd.DataFrame(data_to_save)
    df.to_csv(filename, index=False)
    # print(f'gt file saved at {filename}')

def load_gt_pairs_from_csv(filename='gt_pairs.csv'):
    # print(f'load gt file from {filename}')
    # 读取 CSV 文件并将子细胞列表转换为字典格式
    df = pd.read_csv(filename)
    gt_pairs = {}
    for _, row in df.iterrows():
        # 将每个细胞的子细胞列表从字符串恢复为列表
        children = eval(row['children'])  # 使用 eval 将字符串转换为列表
        gt_pairs[row['cell']] = children
    return gt_pairs

def find_ancestors(tree_path, node):
    ancestors = []
    current = node
    
    # 建立父子关系的字典
    parent_dict = dict(zip(tree_path["son"], tree_path["parent"]))
    
    # 递归查找祖先
    while current in parent_dict:
        parent = parent_dict[current]
        ancestors.append(parent)
        current = parent
    
    return ancestors

# 遍历每个细胞及其可能的父细胞关系，计算准确率
def calculate_cell_accuracy(cell_to_cluster, tree_path, gt_pairs):
    # tree_path.set_index('parent')
    acc_dif, acc_same = 0, 0
    total_num = 0
    for i in gt_pairs.keys():
        if i in cell_to_cluster.keys():
            c_i = cell_to_cluster[i]
            gt_children = gt_pairs[i]
            for child in gt_children:
                if child in cell_to_cluster.keys():
                    c_j = cell_to_cluster[child]
                    # if c_i == c_j:
                    #     continue
                    total_num += 1
                    # if c_i in tree_path.index and c_j in tree_path.loc[c_i].values:
                    if c_i in find_ancestors(tree_path, c_j):
                        acc_dif += 1
                    elif c_i == c_j: # 属于同一簇或父细胞属于子细胞祖先簇
                        acc_same += 1
    return acc_dif, acc_same, total_num

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, default='17')
    parser.add_argument('--method', type=str, default='leiden')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    root_dir = '/home/ubuntu/duxinghao/clone'
    # load cnv data
    id = args.dataset_id
    dataset_name = f'c{id}_CNV'
    # data = pd.read_csv(f'../data/lineage_trace_data/lineage_trace_data/{dataset_name}.csv', index_col=0)
    data = pd.read_csv(f'{root_dir}/data/lineage_trace_data/m5k_lg{id}_character_matrix.alleleThresh.txt', sep="\t", index_col=0)
    data.index = data.index.str.strip() # .str.split('.').str[1]
    data = data.replace('-', np.nan)
    threshold = int(data.shape[1] * 0.7)  # 保留70%非缺失值的行
    data_cleaned = data.dropna(thresh=threshold)
    data_filled = data_cleaned.fillna(-1).astype(int)

    # compute ground truth pairs
    gt_path = f'{root_dir}/data/lineage_trace_data/{dataset_name}_gt_inherit.csv'
    # if not os.path.exists(gt_path) or True:
    #     gt_pairs = calc_gt_pairs(data_filled)
    #     save_gt_pairs_to_csv(gt_pairs, filename=gt_path)
    # else:
    gt_pairs = load_gt_pairs_from_csv(filename=gt_path)

    res_dir = f'{root_dir}/rl_leiden/results/lineage_trace_data/{args.method}/CNV/leiden'
    pred_labels = pd.read_csv(f'{res_dir}/{dataset_name}/cell2cluster.csv')
    tree_path = pd.read_csv(f'{res_dir}/{dataset_name}/tree_path.csv')
    cell_to_cluster = pred_labels.set_index('cell')['cluster']
    cell_to_cluster.index = cell_to_cluster.index.str.strip()

    # comput acc
    acc_dif, acc_same, n_pairs = calculate_cell_accuracy(cell_to_cluster, tree_path, gt_pairs)
    accuracy = (acc_dif+acc_same) / n_pairs
    print(f"{args.method}: Num of GT pairs: {n_pairs} SameCorrect: {acc_same}, DifCorrect: {acc_dif} Prediction Accuracy: {accuracy}")
