"""sparse_snn.py v2 — 可训练稀疏 SNN (完整 D 规律 + 实测 firing rate)。

改进:
1. 掩码用完整 D 规律: 长尾度分布 + E/I 符号 (60/40) 初始化
2. 更大训练规模: 缩小与稠密 MLP 的准确率差距
3. 实测 firing rate (能耗数据做实, 不再假设 0.1)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist():
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
    ytr = d['y_train'].astype(np.int64)
    Xte = d['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
    yte = d['y_test'].astype(np.int64)
    return Xtr, ytr, Xte, yte


def surrogate_spike(v, threshold=1.0):
    spike = (v > threshold).float()
    return spike + torch.sigmoid(v - threshold) - torch.sigmoid(v - threshold).detach()


class SparseSNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, mask, sign, T=15, decay=0.9, threshold=1.0):
        super().__init__()
        self.T = T
        self.decay = decay
        self.threshold = threshold
        # 权重幅度可训练, 符号(E/I)作为初始化先验
        self.w_hid = nn.Parameter(torch.randn(hid_dim, in_dim).abs() * 0.05 * sign)
        self.register_buffer('mask', mask)   # (hid_dim, in_dim) 0/1
        self.w_out = nn.Parameter(torch.randn(out_dim, hid_dim) * 0.05)
        self.hid_dim = hid_dim

    def forward(self, x):
        B = x.shape[0]
        W = self.w_hid * self.mask
        v = torch.zeros(B, self.hid_dim, device=x.device)
        rate = torch.zeros(B, self.hid_dim, device=x.device)
        for _ in range(self.T):
            I = x @ W.T
            v = v * self.decay + I
            s = surrogate_spike(v, self.threshold)
            v = v * (1.0 - s)
            rate = rate + s
        rate = rate / self.T
        return rate @ self.w_out.T, rate.mean()   # (logits, 平均 firing rate)


class DenseMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))), 0.0


def longtail_mask_ei(hid_dim, in_dim, density, e_ratio=0.6, seed=0):
    """长尾度分布 + E/I 符号 (D 规律)。返回 (mask, sign)。"""
    rng = np.random.default_rng(seed)
    avg_deg = density * in_dim
    degs = (rng.pareto(2.0, hid_dim) + 1.0) * (avg_deg / 2.0)
    degs = np.clip(degs, 1, in_dim).astype(int)
    mask = np.zeros((hid_dim, in_dim), dtype=np.float32)
    sign = np.zeros((hid_dim, in_dim), dtype=np.float32)
    for i, d in enumerate(degs):
        cols = rng.choice(in_dim, size=d, replace=False)
        mask[i, cols] = 1.0
        sign[i, cols] = np.where(rng.random(d) < e_ratio, 1.0, -1.0).astype(np.float32)
    return torch.tensor(mask), torch.tensor(sign)


def train_model(model, Xtr, ytr, epochs, batch=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
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
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {ep+1}: loss={total/n:.4f}")


def evaluate(model, Xte, yte):
    model.eval()
    Xt = torch.tensor(Xte, device=device)
    yt = torch.tensor(yte, device=device)
    with torch.no_grad():
        out, fr = model(Xt)
        acc = float((out.argmax(1) == yt).float().mean().item())
        return acc, float(fr)


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    in_dim, hid_dim, out_dim = 784, 800, 10
    n_train = 20000
    epochs = 6

    print("=== 稠密 MLP 基线 ===")
    mlp = DenseMLP(in_dim, hid_dim, out_dim).to(device)
    t0 = time.time()
    train_model(mlp, Xtr[:n_train], ytr[:n_train], epochs)
    mlp_acc, _ = evaluate(mlp, Xte[:2000], yte[:2000])
    print(f"  准确率={mlp_acc:.4f}  用时={time.time()-t0:.0f}s")

    print("\n=== 稀疏 SNN (长尾 + E/I 符号, surrogate gradient) ===")
    results = []
    for density in [0.1, 0.05, 0.02]:
        mask, sign = longtail_mask_ei(hid_dim, in_dim, density)
        snn = SparseSNN(in_dim, hid_dim, out_dim, mask.to(device), sign.to(device)).to(device)
        t0 = time.time()
        train_model(snn, Xtr[:n_train], ytr[:n_train], epochs)
        acc, fr = evaluate(snn, Xte[:2000], yte[:2000])
        sp = float(mask.mean())
        print(f"  密度={density} 稀疏度={sp:.3f}: 准确率={acc:.4f}  firing_rate={fr:.4f}  用时={time.time()-t0:.0f}s")
        results.append((density, acc, sp, fr))

    # 能耗对比 (实测 firing rate)
    print("\n=== 能耗对比 (实测 firing rate, 45nm 模型) ===")
    mac_pj, add_pj = 3.7, 0.9
    flops_mlp = 2 * (in_dim * hid_dim + hid_dim * out_dim)
    e_mlp = flops_mlp * mac_pj
    print(f"  稠密 MLP: {flops_mlp:.1e} FLOPs -> {e_mlp:.0f} pJ")
    for density, acc, sp, fr in results:
        conn = int(in_dim * hid_dim * sp) + hid_dim * out_dim
        sops = conn * fr * 15          # T=15, 实测 firing rate
        e_snn = sops * add_pj
        print(f"  稀疏 SNN (密度{density}): 实测fr={fr:.3f}, {sops:.1e} 加法 -> {e_snn:.0f} pJ, "
              f"比MLP省 {e_mlp/e_snn:.0f}x")


if __name__ == '__main__':
    main()
