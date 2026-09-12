"""递归脉冲网络 — 处理时序输入序列 (SNN 的天然强项)。"""
import torch
import torch.nn as nn
from .neurons import surrogate_spike


class RecurrentSNN(nn.Module):
    """递归脉冲 SNN: 处理时序输入 (B, seq_len, in_dim)。

    每步: I = x[:,t] @ W_in.T + s @ W_rec.T (上一步 spike 递归)
    小世界递归连接提供时序记忆。
    """

    def __init__(self, in_dim, hid_dim, out_dim, mask_in, sign_in, mask_rec, sign_rec,
                 decay=0.9, threshold=1.0, init_scale=0.05):
        super().__init__()
        self.decay = decay
        self.threshold = threshold
        self.hid_dim = hid_dim
        self.w_in = nn.Parameter(torch.randn(hid_dim, in_dim, device=sign_in.device).abs() * init_scale * sign_in)
        self.w_rec = nn.Parameter(torch.randn(hid_dim, hid_dim, device=sign_rec.device).abs() * init_scale * sign_rec)
        self.register_buffer('mask_in', mask_in)     # (hid, in)
        self.register_buffer('mask_rec', mask_rec)   # (hid, hid)
        self.readout = nn.Linear(hid_dim, out_dim)

    @property
    def total_sparsity(self):
        total = self.mask_in.numel() + self.mask_rec.numel()
        active = int(self.mask_in.sum()) + int(self.mask_rec.sum())
        return 1.0 - active / total

    def forward(self, x):
        """x: (B, seq_len, in_dim) -> (logits, mean_firing_rate)。"""
        B, seq_len, _ = x.shape
        Win = self.w_in * self.mask_in
        Wrec = self.w_rec * self.mask_rec
        v = torch.zeros(B, self.hid_dim, device=x.device)
        rate = torch.zeros(B, self.hid_dim, device=x.device)
        s = torch.zeros(B, self.hid_dim, device=x.device)
        for t in range(seq_len):
            I = x[:, t] @ Win.T + s @ Wrec.T
            v = v * self.decay + I
            s = surrogate_spike(v, self.threshold)
            v = v * (1.0 - s)
            rate = rate + s
        rate = rate / seq_len
        return self.readout(rate), rate.mean()
