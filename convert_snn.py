"""convert_snn.py — 转换法: ReLU MLP → IF SNN (firing rate 编码)。

核心: IF 神经元 firing rate ≈ ReLU(输入电流), 用权重归一化让发放率精确复现激活值。
步骤: 训练 ReLU MLP → 统计每层最大激活 → 权重归一化 → IF SNN 跑 T 步。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist():
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
    ytr = d['y_train'].astype(np.int64)
    Xte = d['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
    yte = d['y_test'].astype(np.int64)
    return Xtr, ytr, Xte, yte


class ReLUMLP(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class IFSNN(nn.Module):
    """IF 脉冲网络, 由 ReLU MLP 转换而来。"""

    def __init__(self, mlp, max1, T):
        super().__init__()
        self.T = T
        self.threshold = 1.0
        # 权重归一化: 层1 /max1, 层2 *max1 (补偿 scale)
        self.w1 = nn.Parameter(mlp.fc1.weight.data / max1, requires_grad=False)
        self.b1 = nn.Parameter(mlp.fc1.bias.data / max1, requires_grad=False)
        self.w2 = nn.Parameter(mlp.fc2.weight.data * max1, requires_grad=False)
        self.b2 = nn.Parameter(mlp.fc2.bias.data, requires_grad=False)

    def forward(self, x):
        B = x.shape[0]
        v = torch.zeros(B, self.w1.shape[0], device=x.device)
        spikes = torch.zeros(B, self.w1.shape[0], device=x.device)
        I = x @ self.w1.T + self.b1
        for _ in range(self.T):
            v = v + I                        # IF: 无泄漏累积
            s = (v >= self.threshold).float()
            v = v - s * self.threshold       # 发放后减去阈值
            spikes = spikes + s
        rate = spikes / self.T               # firing rate ≈ ReLU(I)
        return rate @ self.w2.T + self.b2


def train(model, Xtr, ytr, epochs=5, batch=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
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
            opt.step()


def evaluate_ann(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, device=device))
        return float((out.argmax(1) == torch.tensor(yte, device=device)).float().mean())


def evaluate_snn(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, device=device))
        return float((out.argmax(1) == torch.tensor(yte, device=device)).float().mean())


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    in_dim, hid, out_dim = 784, 800, 10
    n_train = 20000

    # 1. 训练 ReLU MLP
    print("=== 1. 训练 ReLU MLP ===")
    mlp = ReLUMLP(in_dim, hid, out_dim).to(device)
    train(mlp, Xtr[:n_train], ytr[:n_train], epochs=5)
    acc_ann = evaluate_ann(mlp, Xte[:2000], yte[:2000])
    print(f"  ANN 准确率={acc_ann:.4f}")

    # 2. 统计最大激活 (归一化用)
    mlp.eval()
    with torch.no_grad():
        h = torch.relu(mlp.fc1(torch.tensor(Xtr[:2000], device=device)))
        max1 = float(h.max().item())
    print(f"  隐藏层最大激活={max1:.3f}")

    # 3. 转换 IF SNN, 不同 T 步
    print("\n=== 2. 转换 IF SNN (firing rate 编码) ===")
    for T in [5, 10, 20, 50, 100]:
        snn = IFSNN(mlp, max1, T).to(device)
        acc_snn = evaluate_snn(snn, Xte[:2000], yte[:2000])
        print(f"  T={T:3d}: SNN 准确率={acc_snn:.4f}  (ANN {acc_ann:.4f})")


if __name__ == '__main__':
    main()
