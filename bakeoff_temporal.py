"""bakeoff_temporal.py — 实验2a: 时序脉冲编码对拍。

MNIST 每个像素强度 → 泊松脉冲序列(rate coding), 储备池在 T 步内积分脉冲。
测拓扑在「时序信息处理」上的差异 —— 大脑本质是时序系统, 这是最本质的测试。
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


def to_spikes(X, T, rng, max_rate=0.5):
    rates = X * max_rate                       # (B, d_in)
    S = rng.random((X.shape[0], X.shape[1], T)) < rates[..., None]
    return S.astype(np.float32)


def calibrate(res, X, T, target_rate=0.1, iters=10, seed=0):
    """调输入 gain, 让平均 firing rate 接近 target_rate。"""
    rng = np.random.default_rng(seed)
    gain = 1.0
    S = torch.tensor(to_spikes(X[:256], T, rng), device=device)
    for _ in range(iters):
        rate = res.run_spikes(S, T, gain).mean().item()
        if rate < 1e-8:
            gain *= 10.0
        else:
            gain *= target_rate / max(rate, 1e-8)
    return float(gain)


def extract_features_spikes(res, X, T, gain, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    feats = []
    for i in range(0, len(X), batch):
        S = torch.tensor(to_spikes(X[i:i + batch], T, rng), device=device)
        feats.append(res.run_spikes(S, T, gain).cpu().numpy())
    return np.concatenate(feats, axis=0)


def ridge_fit(Z, y, n_classes=10, lam=1e-3):
    Y = np.eye(n_classes)[y]
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
    return W


def main():
    A = sp.load_npz('data/malecns_signed_adj.npz')
    N = 3000
    T = 30
    d_in = 784

    A_sub = extract_subgraph(A, N)
    print(f"子图: {A_sub.shape}, nnz={A_sub.nnz}")

    topos = make_topologies(A_sub, seed=42)
    W_in_np = make_W_in(N, d_in)

    Xtr, ytr, Xte, yte = load_mnist(limit_train=20000, limit_test=2000)
    print(f"MNIST: train={len(Xtr)}, test={len(Xte)}, T={T} (时序脉冲编码)")

    results = {}
    for name, Amat in topos.items():
        t0 = time.time()
        W = to_torch_csr(Amat)
        W_in = torch.tensor(W_in_np, device=device)
        res = LIFReservoir(W, W_in)

        gain = calibrate(res, Xtr, T)
        Ztr = extract_features_spikes(res, Xtr, T, gain)
        Wout = ridge_fit(Ztr, ytr)
        Zte = extract_features_spikes(res, Xte, T, gain)
        pred = Zte @ Wout
        acc = float(np.mean(np.argmax(pred, axis=1) == yte))
        fr = float(Zte.mean())

        results[name] = (acc, fr)
        print(f"[{name}] acc={acc:.4f}  firing_rate={fr:.4f}  gain={gain:.2f}  用时={time.time()-t0:.1f}s")

    print("\n=== 时序脉冲对拍结果 ===")
    for name, (acc, fr) in results.items():
        print(f"  {name:8s}  准确率={acc:.4f}  firing_rate={fr:.4f}")


if __name__ == '__main__':
    main()
