"""sequential_mnist.py — 时序任务 (sMNIST 行级): 递归 SNN 的时序能力。

把 MNIST 28×28 图像按行拆成 28 步时序输入, 测试脉冲网络的时序记忆。
对比: 递归 SNN (小世界递归) vs 稠密 GRU 基线。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from awareliquid import RecurrentSNN, longtail_mask, smallworld_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist_seq():
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].astype(np.float32) / 255.0     # (60000, 28, 28)
    ytr = d['y_train'].astype(np.int64)
    Xte = d['x_test'].astype(np.float32) / 255.0
    yte = d['y_test'].astype(np.int64)
    return Xtr, ytr, Xte, yte


class DenseGRU(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, batch_first=True)
        self.fc = nn.Linear(hid, out)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1]), 0.0


def build_recurrent(hid=500, k=20):
    mask_in, sign_in = longtail_mask(hid, 28, 0.5)
    mask_rec, sign_rec = smallworld_mask(hid, k, 0.1)
    return RecurrentSNN(28, hid, 10, mask_in.to(device), sign_in.to(device),
                        mask_rec.to(device), sign_rec.to(device)).to(device)


def train(model, X, y, epochs, batch=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out, _ = model(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {ep+1}: loss={total/n:.4f}")


def evaluate(model, X, y):
    model.eval()
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    with torch.no_grad():
        out, _ = model(Xt)
        return float((out.argmax(1) == yt).float().mean().item())


def main():
    Xtr, ytr, Xte, yte = load_mnist_seq()
    n_train, epochs = 20000, 8

    print("=== 稠密 GRU 基线 (sMNIST 28 步) ===")
    gru = DenseGRU(28, 500, 10).to(device)
    t0 = time.time()
    train(gru, Xtr[:n_train], ytr[:n_train], epochs)
    a_gru = evaluate(gru, Xte[:2000], yte[:2000])
    print(f"  准确率={a_gru:.4f}  用时={time.time()-t0:.0f}s")

    print("\n=== 递归 SNN (小世界递归, sMNIST 28 步, 梯度裁剪 + 15 epochs) ===")
    snn = build_recurrent(hid=500, k=20)
    t0 = time.time()
    train(snn, Xtr[:n_train], ytr[:n_train], 15, lr=5e-4)
    a_snn = evaluate(snn, Xte[:2000], yte[:2000])
    print(f"  准确率={a_snn:.4f}  总稀疏度={snn.total_sparsity:.3f}  用时={time.time()-t0:.0f}s")

    print(f"\n=== 对比 ===")
    print(f"  稠密 GRU: {a_gru:.4f}")
    print(f"  递归 SNN: {a_snn:.4f}  (稀疏度 {snn.total_sparsity:.3f})")
    print(f"  差距: {a_gru - a_snn:.4f} 点")


if __name__ == '__main__':
    main()
