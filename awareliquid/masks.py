"""掩码生成 — 从真实大脑提取的统计规律 (长尾度 + 小世界 + E/I 比例)。"""
import numpy as np
import torch
import networkx as nx


def longtail_mask(out_dim, in_dim, density, e_ratio=0.6, seed=0):
    """长尾(幂律)度分布掩码 + E/I 符号。返回 (mask, sign), 形状 (out_dim, in_dim)。

    D 规律: 果蝇连接组度分布 max/mean≈75 (无标度), E/I = 60/40。
    """
    rng = np.random.default_rng(seed)
    avg_deg = density * in_dim
    degs = (rng.pareto(2.0, out_dim) + 1.0) * (avg_deg / 2.0)
    degs = np.clip(degs, 1, in_dim).astype(int)
    mask = np.zeros((out_dim, in_dim), dtype=np.float32)
    sign = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i, d in enumerate(degs):
        cols = rng.choice(in_dim, size=d, replace=False)
        mask[i, cols] = 1.0
        sign[i, cols] = np.where(rng.random(d) < e_ratio, 1.0, -1.0).astype(np.float32)
    return torch.tensor(mask), torch.tensor(sign)


def smallworld_mask(n, k, p, e_ratio=0.6, seed=0):
    """小世界递归掩码 (Watts-Strogatz: 高聚类 + 短路径) + E/I 符号。返回 (mask, sign), 形状 (n, n)。

    D 规律: 果蝇连接组聚类系数 6.65× 于随机图, 路径长度短。
    """
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    rng = np.random.default_rng(seed)
    A = nx.to_numpy_array(G).astype(np.float32)
    sign = np.zeros_like(A)
    for i, j in G.edges():
        sign[i, j] = 1.0 if rng.random() < e_ratio else -1.0
    return torch.tensor(A), torch.tensor(sign)
