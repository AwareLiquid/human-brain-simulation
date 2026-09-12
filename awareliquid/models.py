"""多层稀疏 SNN 模型。"""
import torch
import torch.nn as nn
from .layers import SparseLIF


class SparseSNN(nn.Module):
    """多层稀疏脉冲 SNN。

    dims:  [in, hid1, hid2, ..., hidL]  各层维度
    masks: 每层 (hid_{i+1}, hid_i) 的稀疏掩码
    signs: 每层的 E/I 符号
    最后一层 hidL 的 firing rate 经线性读出到 num_classes。
    """

    def __init__(self, dims, masks, signs, num_classes, T=15, decay=0.9,
                 threshold=1.0, inter_gain=1.0):
        super().__init__()
        self.T = T
        self.inter_gain = inter_gain
        self.layers = nn.ModuleList([
            SparseLIF(dims[i], dims[i + 1], masks[i], signs[i], T, decay, threshold)
            for i in range(len(dims) - 1)
        ])
        self.readout = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        """x: (B, in) -> (logits, mean_firing_rate)。"""
        frs = []
        for layer in self.layers:
            x = layer(x)
            frs.append(x.mean())
            x = x * self.inter_gain     # 层间增益, 补偿 rate 信号衰减
        logits = self.readout(x)
        return logits, torch.stack(frs).mean()

    def forward_sparse(self, x):
        """推理: 真稀疏 matmul。"""
        for layer in self.layers:
            x = layer.forward_sparse(x)
            x = x * self.inter_gain
        return self.readout(x), x.mean()

    @property
    def total_sparsity(self):
        """整个模型的平均连接稀疏度。"""
        total, active = 0, 0
        for layer in self.layers:
            total += layer.mask.numel()
            active += int(layer.mask.sum().item())
        return 1.0 - active / max(total, 1)
