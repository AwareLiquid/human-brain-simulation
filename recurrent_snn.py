"""recurrent_snn.py — 阶段三: 递归 SNN (小世界递归连接)。

架构: 784 → 隐藏(前馈长尾稀疏) → 隐藏(小世界递归) → 10(读出)
递归连接用小世界图 (Watts-Strogatz: 高聚类 + 短路径), 落地 D 的"小世界"规律。
surrogate gradient 训练 (BPTT through time)。
对比: 稠密 MLP 基线 + 前馈稀疏 SNN。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
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


class RecurrentSNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, mask_ff, sign_ff, mask_rec, sign_rec,
                 T=15, decay=0.9, threshold=1.0):
        super().__init__()
        self.T = T
        self.decay = decay
        self.threshold = threshold
        self.w_ff = nn.Parameter(torch.randn(hid_dim, in_dim).abs() * 0.05 * sign_ff)
        self.w_rec = nn.Parameter(torch.randn(hid_dim, hid_dim).abs() * 0.02 * sign_rec)
        self.register_buffer('mask_ff', mask_ff)
        self.register_buffer('mask_rec', mask_rec)
        self.w_out = nn.Parameter(torch.randn(out_dim, hid_dim) * 0.05)
        self.hid_dim = hid_dim

    def forward(self, x):
        B = x.shape[0]
        Wff = self.w_ff * self.mask_ff
        Wrec = self.w_rec * self.mask_rec
        v = torch.zeros(B, self.hid_dim, device=x.device)
        rate = torch.zeros(B, self.hid_dim, device=x.device)
        s = torch.zeros(B, self.hid_dim, device=x.device)
        for _ in range(self.T):
            I_ff = x @ Wff.T
            I_rec = s @ Wrec.T                       # 上一步 spike 驱动递归
            v = v * self.decay + I_ff + I_rec
            s = surrogate_spike(v, self.threshold)
            v = v * (1.0 - s)
            rate = rate + s
        rate = rate / self.T
        return rate @ self.w_out.T, rate.mean()


class DenseMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))), 0.0


def longtail_mask(hid_dim, in_dim, density, e_ratio=0.6, seed=0):
    """前馈: 长尾度分布 + E/I 符号。"""
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


def smallworld_mask(n, k, p, e_ratio=0.6, seed=0):
    """递归: 小世界图 (Watts-Strogatz, 高聚类+短路径) + E/I 符号。"""
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    rng = np.random.default_rng(seed)
    A = nx.to_numpy_array(G).astype(np.float32)         # (n, n) 0/1
    sign = np.zeros_like(A)
    for i, j in G.edges():
        sign[i, j] = 1.0 if rng.random() < e_ratio else -1.0
    return torch.tensor(A), torch.tensor(sign)


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
        return float((out.argmax(1) == yt).float().mean().item()), float(fr)


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    in_dim, hid_dim, out_dim = 784, 500, 10
    n_train, epochs = 20000, 6

    print("=== 稠密 MLP 基线 ===")
    mlp = DenseMLP(in_dim, hid_dim, out_dim).to(device)
    t0 = time.time()
    train_model(mlp, Xtr[:n_train], ytr[:n_train], epochs)
    mlp_acc, _ = evaluate(mlp, Xte[:2000], yte[:2000])
    print(f"  准确率={mlp_acc:.4f}  用时={time.time()-t0:.0f}s")

    print("\n=== 递归 SNN (长尾前馈 + 小世界递归) ===")
    mask_ff, sign_ff = longtail_mask(hid_dim, in_dim, 0.05)
    results = []
    for k, p in [(10, 0.1), (20, 0.1), (30, 0.1)]:
        mask_rec, sign_rec = smallworld_mask(hid_dim, k, p)
        snn = RecurrentSNN(in_dim, hid_dim, out_dim,
                           mask_ff.to(device), sign_ff.to(device),
                           mask_rec.to(device), sign_rec.to(device)).to(device)
        t0 = time.time()
        train_model(snn, Xtr[:n_train], ytr[:n_train], epochs)
        acc, fr = evaluate(snn, Xte[:2000], yte[:2000])
        rec_sp = float(mask_rec.mean())
        print(f"  递归k={k} (递归稀疏度={rec_sp:.3f}): 准确率={acc:.4f}  firing_rate={fr:.4f}  用时={time.time()-t0:.0f}s")
        results.append((k, acc, fr, rec_sp))

    print("\n=== 汇总 ===")
    print(f"  稠密 MLP: {mlp_acc:.4f}")
    for k, acc, fr, rec_sp in results:
        print(f"  递归 SNN k={k}: {acc:.4f} (firing_rate={fr:.3f})")


if __name__ == '__main__':
    main()
