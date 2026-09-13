"""h01_snn.py — H01 人脑连接组 → LIF 储备池对拍 (人脑拓扑 vs 随机拓扑)。

验证真实人脑连接组 (H01, 16k 神经元) 作为 SNN 储备池的表现。
"""
import numpy as np
import scipy.sparse as sp
import torch
from safetensors.torch import load_file
from engine import to_torch_csr, LIFReservoir, spectral_radius

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_h01():
    m = np.load('data/h01/metadata.npz', allow_pickle=True)
    is_exc = m['is_excitatory']
    e = np.load('data/h01/edges.npz', allow_pickle=True)
    edges = e['edges']
    return is_exc, edges


def build_adjacency(n, edges, inh_gain=4.0):
    pre, post, typ = edges[:, 0], edges[:, 1], edges[:, 2]
    types, counts = np.unique(typ, return_counts=True)
    print(f"  边类型分布: {dict(zip(types.tolist(), counts.tolist()))}")
    # type=1 兴奋(+1), type=2 抑制(-inh_gain)
    sign = np.where(typ == 1, 1.0, -inh_gain).astype(np.float32)
    A = sp.csr_matrix((sign, (pre, post)), shape=(n, n))
    return A


def load_mnist():
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
    ytr = d['y_train'].astype(np.int64)
    Xte = d['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
    yte = d['y_test'].astype(np.int64)
    return Xtr, ytr, Xte, yte


def reservoir_evaluate(A, W_in_np, Xtr, ytr, Xte, yte, T=15):
    W = to_torch_csr(A)
    W_in = torch.tensor(W_in_np, device=device)
    res = LIFReservoir(W, W_in)

    # 校准输入 scale
    scale = 1.0
    x = torch.tensor(Xtr[:256], device=device)
    for _ in range(10):
        rate = res.run(x * scale, T).mean().item()
        if rate < 1e-8:
            scale *= 10.0
        else:
            scale *= 0.1 / max(rate, 1e-8)

    def feats(X):
        X = X * scale
        out = []
        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i:i + 256], device=device)
            out.append(res.run(xb, T).cpu().numpy())
        return np.concatenate(out)

    # 随机投影降维 (16k → 2000), 避免 ridge 解 16087x16087 稠密系统
    proj = np.random.default_rng(0).standard_normal((A.shape[0], 2000)).astype(np.float32)

    Ztr = (feats(Xtr) @ proj).astype(np.float32)
    Y = np.eye(10)[ytr]
    Wout = np.linalg.solve(Ztr.T @ Ztr + 1e-3 * np.eye(2000), Ztr.T @ Y)
    Zte = (feats(Xte) @ proj).astype(np.float32)
    pred = Zte @ Wout
    return float(np.mean(np.argmax(pred, axis=1) == yte)), float(Zte.mean())


def main():
    is_exc, edges = load_h01()
    n = len(is_exc)
    print(f"=== H01 人脑连接组 → SNN 储备池 ===")
    print(f"神经元: {n}, 边: {len(edges)}")

    A = build_adjacency(n, edges)
    print(f"邻接矩阵: {A.shape}, nnz={A.nnz}, 密度={A.nnz/(n*n):.6f}")

    # safetensors 权重分布 (人脑连接组权重规律)
    w = load_file('data/h01/connectome.safetensors')['weights']
    wnz = w[w != 0].abs().numpy()
    print(f"权重(非零): n={len(wnz)}, mean={wnz.mean():.3f}, median={np.median(wnz):.3f}, "
          f"max={wnz.max():.3f}, p99={np.percentile(wnz, 99):.3f}")

    Xtr, ytr, Xte, yte = load_mnist()
    n_train = 5000  # 16k 储备池较慢, 用小训练集

    # W_in: 784 → 16k 随机
    rng = np.random.default_rng(0)
    W_in = rng.standard_normal((n, 784)).astype(np.float32) * 0.05

    print("\n=== 储备池对拍 (MNIST, 5000 训练) ===")
    acc_real, fr_real = reservoir_evaluate(A, W_in, Xtr[:n_train], ytr[:n_train],
                                           Xte[:1000], yte[:1000])
    print(f"  H01 真实拓扑: 准确率={acc_real:.4f}")

    # 多种子 ER 随机对照 (同密度, 同 E/I)
    density = A.nnz / (n * n)
    e_ratio = float(is_exc.mean())
    accs = []
    for seed in [1, 2, 3, 4, 5]:
        rng2 = np.random.default_rng(seed)
        A_er = sp.random(n, n, density=density, format='csr', random_state=seed)
        A_er.data = np.where(rng2.random(A_er.nnz) < e_ratio, 1.0, -4.0).astype(np.float32)
        acc, _ = reservoir_evaluate(A_er, W_in, Xtr[:n_train], ytr[:n_train],
                                    Xte[:1000], yte[:1000])
        accs.append(acc)
        print(f"  ER 随机 (seed={seed}): 准确率={acc:.4f}")
    print(f"\n  ER 均值={np.mean(accs):.4f} ± {np.std(accs):.4f}  vs  H01={acc_real:.4f}")


if __name__ == '__main__':
    main()
