"""bakeoff_plasticity.py — 实验C: 可塑性 (拓扑 × Hebbian 学习协同)。

最后一个拓扑假设: 真实拓扑的模块化/小世界, 是否让 Hebbian 无监督学习
形成更有效的特征结构 (从而学习后分类更好)。
三组拓扑 + 同样的统计 Hebbian 学习, 对比学习前后准确率。
"""
import numpy as np
import torch
import scipy.sparse as sp
import time

from engine import (device, to_torch_csr, extract_subgraph, make_topologies,
                    LIFReservoir, spectral_radius)


def load_mnist(limit_train=None, limit_test=None):
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
    ytr = d['y_train']
    Xte = d['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
    yte = d['y_test']
    if limit_train:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    return Xtr, ytr, Xte, yte


def make_W_in(n, d_in, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d_in)).astype(np.float32)


def calibrate(res, X, T, target_rate=0.1, iters=10):
    scale = 1.0
    x = torch.tensor(X[:256], device=device)
    for _ in range(iters):
        rate = res.run(x * scale, T).mean().item()
        if rate < 1e-8:
            scale *= 10.0
        else:
            scale *= target_rate / max(rate, 1e-8)
    return float(scale)


def extract_features(res, X, scale, T, batch=256):
    X = X * scale
    feats = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], device=device)
        feats.append(res.run(xb, T).cpu().numpy())
    return np.concatenate(feats, axis=0)


def ridge_fit(Z, y, n_classes=10, lam=1e-3):
    Y = np.eye(n_classes)[y]
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
    return W


def evaluate(W_np, W_in_np, Xtr, ytr, Xte, yte, T):
    W = to_torch_csr(W_np)
    W_in = torch.tensor(W_in_np, device=device)
    res = LIFReservoir(W, W_in)
    scale = calibrate(res, Xtr, T)
    Ztr = extract_features(res, Xtr, scale, T)
    Wout = ridge_fit(Ztr, ytr)
    Zte = extract_features(res, Xte, scale, T)
    return float(np.mean(np.argmax(Zte @ Wout, axis=1) == yte))


def hebbian_learn(W_np, X, W_in_np, T, lr=1.0, epochs=2, batch=256):
    """统计 Hebbian 无监督学习: 共发放增强 + 行归一化 + 谱半径归一化。"""
    W = to_torch_csr(W_np)
    W_in = torch.tensor(W_in_np, device=device)
    res = LIFReservoir(W, W_in)
    scale = calibrate(res, X, T)

    A = W_np.tocoo()
    ii, jj = A.row, A.col
    orig_row_sum = np.asarray(np.abs(W_np).sum(axis=1)).ravel()

    for ep in range(epochs):
        co = np.zeros(len(ii))
        n_batches = 0
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch] * scale, device=device)
            rate = res.run(xb, T).cpu().numpy()          # (B, N)
            co += (rate[:, ii] * rate[:, jj]).mean(axis=0)
            n_batches += 1
        co /= n_batches
        # Hebbian: 增强共发放连接 (保持符号方向)
        new_data = A.data + lr * co * np.sign(A.data)
        W2 = W_np.copy()
        W2.data = new_data
        # homeostatic 行归一化 (每行 |w| 总和守恒)
        row_sum_new = np.asarray(np.abs(W2).sum(axis=1)).ravel()
        fac = np.divide(orig_row_sum, row_sum_new,
                        out=np.ones_like(orig_row_sum), where=row_sum_new > 0)
        W2 = (sp.diags(fac) @ W2).tocsr()
        # 谱半径归一化 (保持稳定区)
        r = spectral_radius(W2)
        if r > 0:
            W2 = (W2 * (0.9 / r)).tocsr()
        W_np = W2
        A = W_np.tocoo()
        ii, jj = A.row, A.col
    return W_np


def main():
    A = sp.load_npz('data/malecns_signed_adj.npz')
    N = 3000
    T = 20
    d_in = 784

    A_sub = extract_subgraph(A, N)
    topos = make_topologies(A_sub, seed=42)
    W_in_np = make_W_in(N, d_in)

    Xtr, ytr, Xte, yte = load_mnist(limit_train=20000, limit_test=2000)

    print("=== 实验C: 可塑性 (拓扑 × Hebbian 学习) ===")
    for name, W0 in topos.items():
        t0 = time.time()
        acc_before = evaluate(W0, W_in_np, Xtr, ytr, Xte, yte, T)
        W_learned = hebbian_learn(W0, Xtr, W_in_np, T, epochs=2)
        acc_after = evaluate(W_learned, W_in_np, Xtr, ytr, Xte, yte, T)
        print(f"  {name:8s} 学习前={acc_before:.4f}  学习后={acc_after:.4f}  "
              f"Δ={acc_after-acc_before:+.4f}  用时={time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
