"""spike_recurrent.py — spike-based 循环训练 (转换法初始化 + spike 微调)。

目标: 解决 rate-based 循环转换低效 (2800 步)。用 spike 循环 (每步 1 步, 共 28 步)
+ surrogate gradient 微调, 让网络在高效步数下保持准确率。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from awareliquid import surrogate_spike

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist_seq():
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].astype(np.float32) / 255.0
    ytr = d['y_train'].astype(np.int64)
    Xte = d['x_test'].astype(np.float32) / 255.0
    yte = d['y_test'].astype(np.int64)
    return Xtr, ytr, Xte, yte


class RecurrentReLU(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.wx = nn.Linear(in_dim, hid)
        self.wh = nn.Linear(hid, hid, bias=False)
        self.wo = nn.Linear(hid, out)

    def forward(self, x):
        B, seq, _ = x.shape
        h = torch.zeros(B, self.wx.out_features, device=x.device)
        for t in range(seq):
            h = torch.relu(self.wx(x[:, t]) + self.wh(h))
        return self.wo(h)


class SpikeRecurrent(nn.Module):
    """spike 循环 IF 网络 (每步 1 步), 由循环 ReLU 转换初始化 + 微调。"""

    def __init__(self, in_dim, hid, out, wx, wx_b, wh, wo, wo_b, wh_scale=1.0):
        super().__init__()
        self.wx = nn.Parameter(wx)
        self.wx_b = nn.Parameter(wx_b)
        self.wh = nn.Parameter(wh * wh_scale)      # spike 稀疏, 可能需要更大 scale
        self.wo = nn.Parameter(wo)
        self.wo_b = nn.Parameter(wo_b)
        self.hid = hid

    def forward(self, x):
        B, seq, _ = x.shape
        v = torch.zeros(B, self.hid, device=x.device)
        s = torch.zeros(B, self.hid, device=x.device)
        spikes_total = torch.zeros(B, self.hid, device=x.device)
        for t in range(seq):
            I = x[:, t] @ self.wx.T + self.wx_b + s @ self.wh.T
            v = v + I                            # IF: 无泄漏累积
            s = surrogate_spike(v)               # 发放 (阈值 1)
            v = v - s                            # reset
            spikes_total = spikes_total + s
        rate = spikes_total / seq
        return rate @ self.wo.T + self.wo_b, rate.mean()


def train_rnn(model, X, y, epochs=8, batch=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = model(Xt[idx])
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, device=device))
        if isinstance(out, tuple):
            out = out[0]
        return float((out.argmax(1) == torch.tensor(y, device=device)).float().mean())


def main():
    Xtr, ytr, Xte, yte = load_mnist_seq()
    n_train = 20000
    in_dim, hid, out_dim = 28, 500, 10

    # 1. 训练循环 ReLU
    print("=== 1. 训练循环 ReLU ===")
    rnn = RecurrentReLU(in_dim, hid, out_dim).to(device)
    train_rnn(rnn, Xtr[:n_train], ytr[:n_train], epochs=8)
    acc_ann = evaluate(rnn, Xte[:2000], yte[:2000])
    print(f"  ANN 准确率={acc_ann:.4f}")

    # 2. 统计 99.9 分位数
    rnn.eval()
    with torch.no_grad():
        x = torch.tensor(Xtr[:1000], device=device)
        B, seq, _ = x.shape
        h = torch.zeros(B, hid, device=device)
        all_h = []
        for t in range(seq):
            h = torch.relu(rnn.wx(x[:, t]) + rnn.wh(h))
            all_h.append(h)
        max_h = float(torch.quantile(torch.cat(all_h), 0.999).item())
    print(f"  99.9 分位数={max_h:.3f}")

    # 3. 转换初始化 spike 循环
    wx = (rnn.wx.weight.data / max_h).clone()
    wx_b = (rnn.wx.bias.data / max_h).clone()
    wh = rnn.wh.weight.data.clone()
    wo = (rnn.wo.weight.data * max_h).clone()
    wo_b = rnn.wo.bias.data.clone()

    print("\n=== 2. spike 循环 (28 步, 转换初始化) ===")
    for wh_scale in [1.0, 3.0, 10.0]:
        snn = SpikeRecurrent(in_dim, hid, out_dim, wx, wx_b, wh, wo, wo_b,
                             wh_scale=wh_scale).to(device)
        acc_before = evaluate(snn, Xte[:2000], yte[:2000])
        # 微调
        t0 = time.time()
        train_rnn(snn, Xtr[:n_train], ytr[:n_train], epochs=5, lr=1e-4)
        acc_after = evaluate(snn, Xte[:2000], yte[:2000])
        print(f"  wh_scale={wh_scale}: 转换初始={acc_before:.4f} → 微调后={acc_after:.4f}  "
              f"(ANN {acc_ann:.4f})  用时={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
