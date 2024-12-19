import torch
from sklearn.metrics import silhouette_score as ref_ss

def silhouette_score(X, labels):
    """
    使用PyTorch计算轮廓系数
    Args:
        X (torch.Tensor): 数据矩阵，形状为 (n_samples, n_features)
        labels (torch.Tensor): 每个样本的聚类标签，形状为 (n_samples,)
    Returns:
        float: 平均轮廓系数
    """
    # 数据数量
    n_samples = X.size(0)

    # 计算所有样本之间的欧式距离矩阵
    dist_matrix = torch.cdist(X, X, p=2)  # 使用p=2表示欧氏距离

    # 初始化 a 和 b
    a = torch.zeros(n_samples, device=X.device)
    b = torch.full((n_samples,), float('inf'), device=X.device)

    # 遍历每个簇
    for label in labels.unique():
        # 当前簇的索引
        cluster_mask = (labels == label)
        other_mask = ~cluster_mask
        
        # 当前簇内的距离矩阵（排除自身距离）
        intra_cluster_dist = dist_matrix[cluster_mask][:, cluster_mask]
        a[cluster_mask] = intra_cluster_dist.sum(dim=1) / (cluster_mask.sum() - 1)

        # 计算该簇到其他所有簇的平均距离
        for other_label in labels.unique():
            if label == other_label:
                continue
            other_cluster_mask = (labels == other_label)
            inter_cluster_dist = dist_matrix[cluster_mask][:, other_cluster_mask].mean(dim=1)
            b[cluster_mask] = torch.minimum(b[cluster_mask], inter_cluster_dist)

    # 计算轮廓系数 s(i)
    s = (b - a) / torch.maximum(a, b)
    s[torch.isnan(s)] = 0  # 处理NaN情况（单个样本簇）

    # 返回平均轮廓系数
    return s.mean().item()

# 示例
if __name__ == "__main__":
    # 随机生成数据 (n_samples=10, n_features=2)
    X = torch.tensor([[1.0, 2.0], [1.1, 2.1], [5.0, 5.0], [6.0, 5.5], [5.5, 6.0]], device='cuda')
    labels = torch.tensor([0, 0, 3, 1, 2], device='cuda')  # 两个簇: 0 和 1

    # 计算轮廓系数
    score = silhouette_score(X, labels)

    ref_score = ref_ss(X.detach().cpu().numpy(), labels.detach().cpu().numpy())
    print(f"Silhouette Score: {score:.4f}, ref: {ref_score}")
