"""bakeoff_properties.py — 实验A(鲁棒性) + 实验B(样本效率)。

A: 随机删边, 测重训练后准确率下降曲线 (无标度拓扑理论应更鲁棒)
B: 不同训练样本量, 测学习曲线 (模块化拓扑理论应样本效率更高)
"""
import numpy as np
import torch
import scipy.sparse as sp
import time

from engine import device, to_torch_csr, extract_subgraph, make_topologies, LIFReservoir


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


def evaluate(Amat, W_in_np, Xtr, ytr, Xte, yte, T):
    W = to_torch_csr(Amat)
    W_in = torch.tensor(W_in_np, device=device)
    res = LIFReservoir(W, W_in)
    scale = calibrate(res, Xtr, T)
    Ztr = extract_features(res, Xtr, scale, T)
    Wout = ridge_fit(Ztr, ytr)
    Zte = extract_features(res, Xte, scale, T)
    return float(np.mean(np.argmax(Zte @ Wout, axis=1) == yte))


def delete_edges_fraction(A, f, seed=0):
    rng = np.random.default_rng(seed)
    A = A.tocoo()
    keep = rng.random(A.nnz) > f
    return sp.csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape)


def test_robustness(topos, W_in_np, Xtr, ytr, Xte, yte, T):
    print("\n=== 实验A: 鲁棒性 (随机删边, 重训练) ===")
    fractions = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    header = "  " + "".join(f"{int(f*100):>7d}%" for f in fractions)
    print(header)
    for name in topos:
        row = []
        for f in fractions:
            A_del = delete_edges_fraction(topos[name], f) if f > 0 else topos[name]
            acc = evaluate(A_del, W_in_np, Xtr, ytr, Xte, yte, T)
            row.append(acc)
        print(f"  {name:8s} " + "".join(f"{a:7.3f}" for a in row))


def test_sample_efficiency(topos, W_in_np, Xtr, ytr, Xte, yte, T):
    print("\n=== 实验B: 样本效率 (不同训练量) ===")
    sizes = [500, 1000, 2000, 5000, 10000, 20000]
    header = "  " + "".join(f"{s:>7d}" for s in sizes)
    print(header)
    for name in topos:
        row = []
        for s in sizes:
            acc = evaluate(topos[name], W_in_np, Xtr[:s], ytr[:s], Xte, yte, T)
            row.append(acc)
        print(f"  {name:8s} " + "".join(f"{a:7.3f}" for a in row))


def main():
    A = sp.load_npz('data/malecns_signed_adj.npz')
    N = 3000
    T = 20
    d_in = 784

    A_sub = extract_subgraph(A, N)
    topos = make_topologies(A_sub, seed=42)
    W_in_np = make_W_in(N, d_in)

    Xtr, ytr, Xte, yte = load_mnist(limit_train=20000, limit_test=2000)
    print(f"MNIST: train={len(Xtr)}, test={len(Xte)}")

    t0 = time.time()
    test_robustness(topos, W_in_np, Xtr, ytr, Xte, yte, T)
    test_sample_efficiency(topos, W_in_np, Xtr, ytr, Xte, yte, T)
    print(f"\n总用时: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
