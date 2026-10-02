"""ablation_laws.py — 连接组规律消融: 长尾度分布 vs 均匀度, E/I 平衡 vs 全兴奋。

与主结果 (sparse_snn.py, 90.50% @ density 0.05) 严格同架构, 只改掩码生成方式。
证伪判据: 长尾 和 E/I 各自带来 ≥1 点的提升(超 3 种子标准差)。
"""
import numpy as np
import torch
from sparse_snn import (load_mnist, SparseSNN, longtail_mask_ei,
                        train_model, evaluate, device)


def uniform_mask_ei(hid_dim, in_dim, density, e_ratio=0.6, seed=0):
    """均匀度分布(每行 Poisson 度) + E/I。与 longtail_mask_ei 唯一区别: 度分布形状。"""
    rng = np.random.default_rng(seed)
    avg_deg = density * in_dim
    mask = np.zeros((hid_dim, in_dim), dtype=np.float32)
    sign = np.zeros((hid_dim, in_dim), dtype=np.float32)
    for i in range(hid_dim):
        d = int(np.clip(rng.poisson(avg_deg), 1, in_dim))
        cols = rng.choice(in_dim, size=d, replace=False)
        mask[i, cols] = 1.0
        sign[i, cols] = np.where(rng.random(d) < e_ratio, 1.0, -1.0).astype(np.float32)
    return torch.tensor(mask), torch.tensor(sign)


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    in_dim, hid_dim, out_dim = 784, 800, 10
    n_train, epochs, density = 60000, 6, 0.05

    # 四组: (名称, 掩码函数, E/I 比例)
    configs = [
        ("A 长尾+E/I (主结果)", longtail_mask_ei, 0.6),
        ("B 均匀+E/I",           uniform_mask_ei, 0.6),
        ("C 长尾+全兴奋",         longtail_mask_ei, 1.0),
        ("D 均匀+全兴奋",         uniform_mask_ei, 1.0),
    ]

    print("=== 连接组规律消融 (MNIST, 20000 样本, 6 epochs, density 0.05, 3 种子) ===")
    results = []
    for name, mask_fn, e_ratio in configs:
        accs, frs = [], []
        for seed in list(range(10)):
            mask, sign = mask_fn(hid_dim, in_dim, density, e_ratio=e_ratio, seed=seed)
            snn = SparseSNN(in_dim, hid_dim, out_dim, mask.to(device), sign.to(device)).to(device)
            train_model(snn, Xtr[:n_train], ytr[:n_train], epochs)
            acc, fr = evaluate(snn, Xte[:10000], yte[:10000])
            accs.append(acc); frs.append(fr)
        m, s = np.mean(accs), np.std(accs)
        results.append((name, m, s, np.mean(frs)))
        print(f"  {name}: {m:.4f} ± {s:.4f}  (fr={np.mean(frs):.4f})", flush=True)

    # 归因分解
    print("\n=== 归因分解 ===")
    a, b, c, d = [r[1] for r in results]
    print(f"  长尾贡献 (A-B, E/I 固定): {a-b:+.4f}")
    print(f"  长尾贡献 (C-D, 全兴奋):   {c-d:+.4f}")
    print(f"  E/I 贡献  (A-C, 长尾固定): {a-c:+.4f}")
    print(f"  E/I 贡献  (B-D, 均匀固定): {b-d:+.4f}")


if __name__ == '__main__':
    main()
