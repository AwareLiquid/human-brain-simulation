"""h01_attribution.py — 归因实验: H01 的 3 点优势来自 E/I 增益还是拓扑本身?

四组对照: {H01拓扑, ER随机} × {有增益(抑制-4), 无增益(抑制-1)}
"""
import numpy as np
import scipy.sparse as sp
from h01_snn import load_h01, load_mnist, reservoir_evaluate


def main():
    is_exc, edges = load_h01()
    n = len(is_exc)
    pre, post, typ = edges[:, 0], edges[:, 1], edges[:, 2]

    Xtr, ytr, Xte, yte = load_mnist()
    n_train = 5000
    rng = np.random.default_rng(0)
    W_in = rng.standard_normal((n, 784)).astype(np.float32) * 0.05

    density = len(edges) / (n * n)
    e_ratio = float(is_exc.mean())

    def make_h01(gain):
        sign = np.where(typ == 1, 1.0, -gain).astype(np.float32)
        return sp.csr_matrix((sign, (pre, post)), shape=(n, n))

    def make_er(gain, seed=1):
        r = np.random.default_rng(seed)
        A = sp.random(n, n, density=density, format='csr', random_state=seed)
        A.data = np.where(r.random(A.nnz) < e_ratio, 1.0, -gain).astype(np.float32)
        return A

    print("=== 归因实验 (MNIST, 5000 训练, 单种子) ===")
    combos = [
        ("H01 拓扑 + 增益(-4)", make_h01(4.0)),
        ("H01 拓扑 + 无增益(-1)", make_h01(1.0)),
        ("ER 随机 + 增益(-4)", make_er(4.0)),
        ("ER 随机 + 无增益(-1)", make_er(1.0)),
    ]
    for name, A in combos:
        acc, _ = reservoir_evaluate(A, W_in, Xtr[:n_train], ytr[:n_train],
                                    Xte[:1000], yte[:1000])
        print(f"  {name:24s} 准确率={acc:.4f}")


if __name__ == '__main__':
    main()
