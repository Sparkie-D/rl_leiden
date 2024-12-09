import torch
import argparse
from tqdm import tqdm

from net import *
from utils import *
from cluster import *

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from train import get_args

if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)

    # init data & logger
    set_seed_everywhere(args.seed)
    # data, c2cl, dataset_name = load_data(args.data_dir, args.dataset_id)    
    dataset_name = f'{args.real_ds}_CNV'
    data = pd.read_csv(f'/home/ubuntu/duxinghao/clone/data/lineage_trace_data/lineage_trace_data/{dataset_name}.csv', index_col=0)
    c2cl=None

    log_dir = os.path.join(args.output_dir, args.algo, 'lineage_trace', dataset_name)
    
    # init model & optimizer
    data_size = data.values.shape[0]
    input_dim = data.values.shape[1]
    model = MLP(input_dim, args.hidden_dims, args.n_components*2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # get algorithm
    rl_cluster = RLCluster(args.algo, data, c2cl, args.n_components, model, optimizer, log_dir, device, eval=False)

    # learn
    pred_labels = rl_cluster.learn(args.bc_epochs, args.rl_epochs, args.samples_per_epoch, args.epsilon)

    label_df = pd.DataFrame({'cell':data.index, 'cluster':pred_labels})
    label_df.to_csv(os.path.join(log_dir, 'cell2cluster.csv'), index=None)
    # make tree
    root = maketree(cnv=data.values, labels=pred_labels, dist_func=l2_distance)
    # showtree(root)
    drawtree(root, os.path.join(log_dir, 'tree.png'))
    tree_df = pd.DataFrame(data=get_parent_child_pairs(root), columns=['parent', 'son'])
    tree_df.to_csv(os.path.join(log_dir, 'tree_path.csv'), index=None)

