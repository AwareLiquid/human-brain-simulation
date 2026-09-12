"""fashion_snn.py — 方向1: Fashion-MNIST (更难任务) 对拍。

验证稀疏脉冲 SNN 的效率优势在更难任务上是否成立。
Fashion-MNIST 也是 784 维(28x28灰度), 但分类难度显著高于 MNIST。
"""
import gzip
import numpy as np
import torch
import time
from sparse_snn import (SparseSNN, DenseMLP, longtail_mask_ei, train_model, evaluate, device)


def load_fashion():
    def imgs(p):
        with gzip.open(p, 'rb') as f:
            f.read(4)
            n = int.from_bytes(f.read(4), 'big')
            r = int.from_bytes(f.read(4), 'big')
            c = int.from_bytes(f.read(4), 'big')
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)

    def labels(p):
        with gzip.open(p, 'rb') as f:
            f.read(4)
            n = int.from_bytes(f.read(4), 'big')
            return np.frombuffer(f.read(), dtype=np.uint8)

    d = 'data/fashion/'
    Xtr = imgs(d + 'train-images-idx3-ubyte.gz').astype(np.float32) / 255.0
    ytr = labels(d + 'train-labels-idx1-ubyte.gz').astype(np.int64)
    Xte = imgs(d + 't10k-images-idx3-ubyte.gz').astype(np.float32) / 255.0
    yte = labels(d + 't10k-labels-idx1-ubyte.gz').astype(np.int64)
    return Xtr, ytr, Xte, yte


def main():
    Xtr, ytr, Xte, yte = load_fashion()
    in_dim, hid_dim, out_dim = 784, 800, 10
    epochs = 12

    print("=== 稠密 MLP (Fashion-MNIST 全量) ===")
    mlp = DenseMLP(in_dim, hid_dim, out_dim).to(device)
    t0 = time.time()
    train_model(mlp, Xtr, ytr, epochs)
    mlp_acc, _ = evaluate(mlp, Xte, yte)
    print(f"  准确率={mlp_acc:.4f}  用时={time.time()-t0:.0f}s")

    print("\n=== 稀疏 SNN 密度5% (Fashion-MNIST 全量) ===")
    mask, sign = longtail_mask_ei(hid_dim, in_dim, 0.05)
    snn = SparseSNN(in_dim, hid_dim, out_dim, mask.to(device), sign.to(device)).to(device)
    t0 = time.time()
    train_model(snn, Xtr, ytr, epochs)
    acc, fr = evaluate(snn, Xte, yte)
    print(f"  准确率={acc:.4f}  firing_rate={fr:.4f}  用时={time.time()-t0:.0f}s")

    gap = mlp_acc - acc
    flops_mlp = 2 * (in_dim * hid_dim + hid_dim * out_dim)
    conn = int(in_dim * hid_dim * float(mask.mean())) + hid_dim * out_dim
    sops = conn * fr * 15
    saving = (flops_mlp * 3.7) / (sops * 0.9)
    print(f"\nFashion-MNIST 差距: {gap:.4f} 点 | 能耗省 {saving:.0f}x")


if __name__ == '__main__':
    main()
