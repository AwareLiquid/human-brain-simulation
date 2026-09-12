"""recurrent_conversion.py — 转换法推广到循环网络: 循环 ReLU → IF 脉冲循环。

补 sMNIST 时序短板 (从零训练递归 SNN 只有 ~70%, 转换法应保留循环网络的准确率)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist_seq():
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].astype(np.float32) / 255.0     # (60000, 28, 28)
    ytr = d['y_train'].astype(np.int64)
    Xte = d['x_test'].astype(np.float32) / 255.0
    yte = d['y_test'].astype(np.int64)
    return Xtr, ytr, Xte, yte


class RecurrentReLU(nn.Module):
    """循环 ReLU 网络: h_t = ReLU(Wx x_t + Wh h_{t-1} + b)。"""

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


class RecurrentIF(nn.Module):
    """IF 脉冲循环, 由循环 ReLU 转换而来 (rate-based 循环)。"""

    def __init__(self, rnn, max_h, T_sub=5):
        super().__init__()
        self.T_sub = T_sub
        self.thr = 1.0
        self.wx = nn.Parameter(rnn.wx.weight.data / max_h, requires_grad=False)
        self.wx_b = nn.Parameter(rnn.wx.bias.data / max_h, requires_grad=False)
        # 递归权重不归一化: 输入 r 已是 h/max_h, 再除会变成 h/max_h^2 (scale 错误)
        self.wh = nn.Parameter(rnn.wh.weight.data, requires_grad=False)
        self.wo = nn.Parameter(rnn.wo.weight.data * max_h, requires_grad=False)
        self.wo_b = nn.Parameter(rnn.wo.bias.data, requires_grad=False)

    def forward(self, x):
        B, seq, _ = x.shape
        hid = self.wx.shape[0]
        r = torch.zeros(B, hid, device=x.device)   # firing rate (循环用)
        for t in range(seq):
            I = x[:, t] @ self.wx.T + self.wx_b + r @ self.wh.T
            # 每步跑 T_sub 子步让 firing rate 收敛
            v = torch.zeros(B, hid, device=x.device)
            spikes = torch.zeros(B, hid, device=x.device)
            for _ in range(self.T_sub):
                v = v + I
                s = (v >= self.thr).float()
                v = v - s * self.thr
                spikes = spikes + s
            r = spikes / self.T_sub
        return r @ self.wo.T + self.wo_b


def train(model, X, y, epochs=8, batch=128, lr=1e-3):
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
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, device=device))
        return float((out.argmax(1) == torch.tensor(y, device=device)).float().mean())


def main():
    Xtr, ytr, Xte, yte = load_mnist_seq()
    n_train = 20000
    in_dim, hid, out_dim = 28, 500, 10

    print("=== 1. 训练循环 ReLU (sMNIST 28 步) ===")
    rnn = RecurrentReLU(in_dim, hid, out_dim).to(device)
    t0 = time.time()
    train(rnn, Xtr[:n_train], ytr[:n_train], epochs=8)
    acc_ann = evaluate(rnn, Xte[:2000], yte[:2000])
    print(f"  ANN 准确率={acc_ann:.4f}  用时={time.time()-t0:.0f}s")

    # 统计最大激活 (用 99.9 分位数, 避免极端值把整体压小)
    rnn.eval()
    with torch.no_grad():
        x = torch.tensor(Xtr[:1000], device=device)
        B, seq, _ = x.shape
        h = torch.zeros(B, hid, device=device)
        all_h = []
        for t in range(seq):
            h = torch.relu(rnn.wx(x[:, t]) + rnn.wh(h))
            all_h.append(h)
        all_h = torch.cat(all_h)
        max_h = float(torch.quantile(all_h, 0.999).item())
    print(f"  隐藏层激活 99.9 分位数={max_h:.3f}")

    print("\n=== 2. 转换 IF 脉冲循环 ===")
    for T_sub in [20, 50, 100]:
        snn = RecurrentIF(rnn, max_h, T_sub).to(device)
        acc_snn = evaluate(snn, Xte[:500], yte[:500])
        print(f"  T_sub={T_sub:3d}: SNN 准确率={acc_snn:.4f}  (ANN {acc_ann:.4f})")


if __name__ == '__main__':
    main()
