import torch
import argparse
from tqdm import tqdm

from net import *
from utils import *
from leiden import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/data_large2/')
    parser.add_argument('--dataset_id', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/1127')
    parser.add_argument('--n_components', type=int, default=10, help='pca components.')
    parser.add_argument('--hidden_dims', type=int, default=[512, 512], help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--rl_epochs', type=int, default=10, help='Number of epochs to train E.')
    parser.add_argument('--bc_epochs', type=int, default=100, help='Number of epochs to train bc.')
    parser.add_argument('--samples_per_epoch', type=int, default=50, help='Replay buffer size')
    parser.add_argument('--alpha', type=float, default=1, help='rl weight')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)

    # init data & logger
    set_seed_everywhere(args.seed)
    data, c2cl, dataset_name = load_data(args.data_dir, args.dataset_id)
    log_dir = os.path.join(args.output_dir, dataset_name)
    
    # init model & optimizer
    data_size = data.values.shape[0]
    input_dim = data.values.shape[1]
    model = AEModel(input_dim, args.hidden_dims, args.n_components*2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # get algorithm
    rl_leiden = RLLeiden(data, c2cl, args.n_components, model, optimizer, log_dir, device)

    # learn
    rl_leiden.learn(args.bc_epochs, args.rl_epochs, args.samples_per_epoch)