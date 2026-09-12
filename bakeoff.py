"""bakeoff.py — 三组拓扑对拍: MaleCNS真实 vs ER随机 vs MoE式结构化稀疏。

同密度 / 同 E-I 比例 / 同谱半径, 只隔离「连接结构」变量。
任务: MNIST 分类 (储备池 + 线性读出)。指标: 准确率 + 平均 firing rate (能耗代理)。
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
    """调输入缩放, 让平均 firing rate 接近 target_rate。"""
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


def main():
    A = sp.load_npz('data/malecns_signed_adj.npz')
    body_ids = np.load('data/malecns_body_ids.npy')

    N = 3000
    T = 20
    d_in = 784

    A_sub = extract_subgraph(A, N)
    print(f"子图: {A_sub.shape}, nnz={A_sub.nnz}, 密度={A_sub.nnz/(N*N):.5f}")

    topos = make_topologies(A_sub, seed=42)
    W_in_np = make_W_in(N, d_in)

    Xtr, ytr, Xte, yte = load_mnist(limit_train=20000, limit_test=2000)
    print(f"MNIST: train={len(Xtr)}, test={len(Xte)}")

    results = {}
    for name, Amat in topos.items():
        t0 = time.time()
        W = to_torch_csr(Amat)
        W_in = torch.tensor(W_in_np, device=device)
        res = LIFReservoir(W, W_in)

        scale = calibrate(res, Xtr, T)
        Ztr = extract_features(res, Xtr, scale, T)
        Wout = ridge_fit(Ztr, ytr)
        Zte = extract_features(res, Xte, scale, T)
        pred = Zte @ Wout
        acc = float(np.mean(np.argmax(pred, axis=1) == yte))
        fr = float(Zte.mean())

        results[name] = (acc, fr)
        print(f"[{name}] acc={acc:.4f}  firing_rate={fr:.4f}  用时={time.time()-t0:.1f}s")

    print("\n=== 对拍结果 ===")
    for name, (acc, fr) in results.items():
        print(f"  {name:8s}  准确率={acc:.4f}  firing_rate={fr:.4f}")


if __name__ == '__main__':
    main()
