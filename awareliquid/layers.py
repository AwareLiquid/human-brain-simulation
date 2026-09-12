"""稀疏 LIF 层 — 训练用 dense(mask后), 推理可选真稀疏 matmul。"""
import torch
import torch.nn as nn
from .neurons import surrogate_spike


class SparseLIF(nn.Module):
    """稀疏连接 + LIF 脉冲神经元。

    - 训练: W = w * mask (dense), x @ W.T (dense matmul) —— 梯度正常反传
    - 推理: W 转 CSR, torch.sparse.mm (跳过零元素) —— 真正省算力
    """

    def __init__(self, in_dim, out_dim, mask, sign, T, decay=0.9, threshold=1.0,
                 init_scale=0.05):
        super().__init__()
        self.T = T
        self.decay = decay
        self.threshold = threshold
        self.out_dim = out_dim
        # 权重幅度可训练, 符号(E/I)作为初始化先验
        self.w = nn.Parameter(torch.randn(out_dim, in_dim).abs() * init_scale * sign)
        self.register_buffer('mask', mask)          # (out_dim, in_dim) 0/1

    @property
    def sparsity(self):
        return float(1.0 - self.mask.mean().item())

    def forward(self, x):
        """x: (B, in) 输入电流 -> (B, out) firing rate。dense 路径(训练)。"""
        W = self.w * self.mask
        B = x.shape[0]
        v = torch.zeros(B, self.out_dim, device=x.device)
        rate = torch.zeros(B, self.out_dim, device=x.device)
        for _ in range(self.T):
            I = x @ W.T
            v = v * self.decay + I
            s = surrogate_spike(v, self.threshold)
            v = v * (1.0 - s)
            rate = rate + s
        return rate / self.T

    @torch.no_grad()
    def forward_sparse(self, x):
        """x: (B, in) -> (B, out) firing rate。真稀疏 matmul 路径(推理)。"""
        W = (self.w * self.mask).to_sparse_csr()
        B = x.shape[0]
        v = torch.zeros(B, self.out_dim, device=x.device)
        rate = torch.zeros(B, self.out_dim, device=x.device)
        for _ in range(self.T):
            I = torch.sparse.mm(W, x.T).T       # sparse (out,in) @ (in,B) -> (out,B)
            v = v * self.decay + I
            s = (v > self.threshold).float()    # 推理用硬阈值, 无梯度
            v = v * (1.0 - s)
            rate = rate + s
        return rate / self.T
