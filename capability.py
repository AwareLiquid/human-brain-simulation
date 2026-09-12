"""capability.py — 稀疏 SNN 能力上限 (更大训练规模)。

稠密 MLP vs 稀疏 SNN(密度5%), 都用更大训练(epochs=10, train=30000, hid=800)。
回答: 稀疏 SNN 的准确率上限能否追近稠密 MLP (缩小那 ~5 个点的差距)。
"""
import torch
import time
from sparse_snn import (load_mnist, SparseSNN, DenseMLP, longtail_mask_ei,
                        train_model, evaluate, device)


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    in_dim, hid_dim, out_dim = 784, 800, 10
    n_train, epochs = 30000, 10

    print("=== 稠密 MLP (更大训练) ===")
    mlp = DenseMLP(in_dim, hid_dim, out_dim).to(device)
    t0 = time.time()
    train_model(mlp, Xtr[:n_train], ytr[:n_train], epochs)
    mlp_acc, _ = evaluate(mlp, Xte[:2000], yte[:2000])
    print(f"  准确率={mlp_acc:.4f}  用时={time.time()-t0:.0f}s")

    print("\n=== 稀疏 SNN 密度5% (更大训练) ===")
    mask, sign = longtail_mask_ei(hid_dim, in_dim, 0.05)
    snn = SparseSNN(in_dim, hid_dim, out_dim, mask.to(device), sign.to(device)).to(device)
    t0 = time.time()
    train_model(snn, Xtr[:n_train], ytr[:n_train], epochs)
    acc, fr = evaluate(snn, Xte[:2000], yte[:2000])
    print(f"  准确率={acc:.4f}  firing_rate={fr:.4f}  用时={time.time()-t0:.0f}s")

    gap = mlp_acc - acc
    flops_mlp = 2 * (in_dim * hid_dim + hid_dim * out_dim)
    conn = int(in_dim * hid_dim * float(mask.mean())) + hid_dim * out_dim
    sops = conn * fr * 15
    saving = (flops_mlp * 3.7) / (sops * 0.9)
    print(f"\n差距: {gap:.4f} 个百分点 | 能耗省 {saving:.0f}x")


if __name__ == '__main__':
    main()
