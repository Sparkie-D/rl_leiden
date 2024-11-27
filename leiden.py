import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import anndata as ann
import scanpy as sc
import networkx as nx
from tqdm import tqdm
from utils import cal1B, cal2, cal3
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import v_measure_score as vm_score
from torch.utils.tensorboard import SummaryWriter


class Normalizer(object):
    def __init__(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0) + 1e-5

    def normalize(self, data):
        data = (data - self.mean) / self.std
        return data

    def unnormalize(self, data):
        data = data * self.std + self.mean
        return data
    

class RLLeiden(object):
    def __init__(self, 
                 data, 
                 c2cl, 
                 n_comps,
                 model,
                 optim,
                 log_dir,
                 device
        ):
        self.adata = ann.AnnData(data.values, obs=np.arange(data.shape[0]), var=np.arange(data.shape[1]), dtype=float)
        self.c2cl = c2cl
        cl2idx = {x:i for i, x in enumerate(set(self.c2cl.clone))}
        self.true_labels = self.c2cl.clone.map(cl2idx).values
        self.cnv_data = data.values
        os.makedirs(f'{log_dir}', exist_ok=True)

        self.log_dir = log_dir
        self.logger = SummaryWriter(f'{log_dir}')
        self.device = device

        sc.tl.pca(self.adata, n_comps=n_comps)
        self.pca_results = self.adata.obsm['X_pca']
        self.pca_results = torch.from_numpy(self.pca_results).float().to(device)
        self.pca_normalizer = Normalizer(self.pca_results)
        # self.pca_results = (self.pca_results - self.pca_results.mean(0)) / (self.pca_results.std(0) + 1e-5)
        self.data = torch.from_numpy(data.values).float().to(device)
        self.data = (self.data - self.data.mean(0)) / (self.data.std(0) + 1e-5)
        self.model = model
        self.optim = optim

    def leiden(self, embed):
        if isinstance(embed, torch.Tensor):
            embed = embed.detach().cpu().numpy()
        adata = ann.AnnData(embed, obs=np.arange(embed.shape[0]), var=np.arange(embed.shape[1]))
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        pred_labels = adata.obs['leiden'].astype(int)
        return adata, pred_labels

    def get_reward(self, adata, leiden_labels):
        adjacency_matrix = adata.obsp['connectivities'].toarray()
        graph = nx.from_numpy_array(adjacency_matrix)
        communities = []
        for label in np.unique(leiden_labels):
            community = list(np.where(leiden_labels == label)[0])
            communities.append(community)
        modularity = nx.algorithms.community.modularity(graph, communities)
        return modularity
    
    def get_dist(self):
        z = self.model(self.data)
        z_mean, z_logstd = torch.chunk(z, 2, -1)
        z_std = torch.clamp(z_logstd, min=-1, max=5).exp()
        z_dist = torch.distributions.Normal(z_mean, z_std)

        return z_dist
    
    def evaluate(self, pred_labels):
        nmi = nmi_score(self.true_labels, pred_labels, average_method='arithmetic')
        ari = ari_score(self.true_labels, pred_labels)
        vm = vm_score(self.true_labels, pred_labels)
        sc1b = cal1B(truth=len(set(self.true_labels)), pred=len(set(pred_labels)))
        sc2 = cal2(truth=self.true_labels, pred=pred_labels)
        sc3 = cal3(truth=self.true_labels, pred=pred_labels, cnv=self.cnv_data)
        return {'nmi': nmi, 'ari': ari, 'vm': vm, 'sc1b': sc1b, 'sc2': sc2, 'sc3': sc3}

    def learn(self, bc_epochs, rl_epochs, samples_per_epoch, alpha=1):
        best_adata, best_labels = self.leiden(self.pca_results.detach().cpu().numpy())
        best_return = self.get_reward(best_adata, best_labels)
        baseline_return = best_return
        normalize_pca = self.pca_normalizer.normalize(self.pca_results)

        results = self.evaluate(best_labels.values)
        df = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
        df.to_csv(os.path.join(self.log_dir, 'result.csv'))
        
        for e in tqdm(range(bc_epochs), desc='BC Training Epochs'):
            z_dist = self.get_dist()
            z_sample = z_dist.rsample()
            bc_loss = -z_dist.log_prob(normalize_pca).mean()
            decode = self.model.decode(z_dist.mean)
            reconstruc_loss = torch.abs(decode - self.data).mean()

            loss = bc_loss + reconstruc_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.logger.add_scalar('train/init_bc_loss', bc_loss.item(), e)

        for e in tqdm(range(rl_epochs), desc='Training Epochs'):
            z_dist = self.get_dist()
            log_probs, rs = [], []
            for i in range(samples_per_epoch):
                # sample data
                z_sample = z_dist.rsample()

                # get return
                adata, labels = self.leiden(self.pca_normalizer.unnormalize(z_sample).detach().cpu().numpy())
                r = self.get_reward(adata, labels)
                log_probs.append(z_dist.log_prob(z_sample).mean())
                rs.append(r)

                if r > best_return:
                    best_return = r
                    best_adata = adata
                    best_labels = labels

            rs = torch.Tensor(rs).to(z_sample)
            rl_loss = 0
            for i in range(samples_per_epoch):
                rl_loss -= (rs[i]- rs.mean()) * log_probs[i]
            # bc_loss = -z_dist.log_prob(normalize_pca).mean()
            # loss = 0 * bc_loss + alpha * rl_loss

            self.optim.zero_grad()
            rl_loss.backward()
            self.optim.step()

            results = self.evaluate(best_labels.values)
            for k, v in results.items():
                self.logger.add_scalar(f'eval/{k}', v, e)
            df = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
            df.to_csv(os.path.join(self.log_dir, 'result.csv'))

            # self.logger.add_scalar('train/loss', loss.item(), e)
            # self.logger.add_scalar('train/bc_loss', bc_loss.item(), e)
            self.logger.add_scalar('train/rl_loss', rl_loss.item(), e)
            self.logger.add_scalar('train/best_return', best_return, e)
            self.logger.add_scalar('train/return', rs.mean().item(), e)



