import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

from net import *
from utils import *
from cluster import *

data_dir = '/home/ubuntu/duxinghao/clone/data/CNV_multiSample_martix'
ds = 'data2_CNV'
log_dir = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/CNV_multiSample_martix/rl_leiden/CNV/leiden/{ds}'

data, c2cl, dataset_name, non_epis = load_real_data(data_dir, ds, remove_no_epi=False)
df = pd.read_csv(f'{log_dir}/tree_path.csv')
pred_labels = pd.read_csv(f'{log_dir}/cell2cluster.csv', index_col=0).values.reshape(-1)

root = maketree(cnv=data.values, labels=pred_labels, dist_func=l2_distance)
# showtree(root)
tree_df = pd.DataFrame(data=get_parent_child_pairs(root), columns=['parent', 'son'])
# tree_df.to_csv(os.path.join(log_dir, 'tree_path.csv'), index=None)
drawtree(tree_df, os.path.join(log_dir, 'tree_new.pdf'))

# os.remove(f'{dir}/tree.png')

