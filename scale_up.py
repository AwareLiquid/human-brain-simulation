"""scale_up.py — 更大规模 SNN (2层2000 / 2层2500 / 3层2000)。

在 scale_benchmark 基础上继续扩大, 看 MNIST 规模收益是否持续。
"""
import time
from benchmark import build_model, train, evaluate, load_mnist


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    n_train, epochs = 20000, 8

    print("=== 更大规模 SNN (MNIST, 20000 样本, 8 epochs, 稀疏度 5%) ===")
    for dims, label in [
        ([784, 2000, 2000], "2 层 · 2000 隐"),
        ([784, 2500, 2500], "2 层 · 2500 隐"),
        ([784, 2000, 2000, 2000], "3 层 · 2000 隐"),
    ]:
        m = build_model(dims, 0.05, inter_gain=5.0)
        t0 = time.time()
        train(m, Xtr[:n_train], ytr[:n_train], epochs)
        acc = evaluate(m, Xte[:2000], yte[:2000])
        print(f"  {label:16s} 准确率={acc:.4f}  稀疏度={m.total_sparsity:.3f}  用时={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
