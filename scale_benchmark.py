"""scale_benchmark.py — 深度/规模收益曲线: 1/2/3 层稀疏 SNN + 更大隐藏层。"""
import time
from benchmark import build_model, train, evaluate, load_mnist


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    n_train, epochs = 20000, 8

    configs = [
        ([784, 800], "1 层 · 800 隐"),
        ([784, 800, 800], "2 层 · 800 隐"),
        ([784, 800, 800, 800], "3 层 · 800 隐"),
        ([784, 1500, 1500], "2 层 · 1500 隐"),
    ]

    print("=== 深度/规模收益曲线 (MNIST, 20000 样本, 8 epochs, 稀疏度 5%) ===")
    for dims, label in configs:
        m = build_model(dims, 0.05, inter_gain=5.0)
        t0 = time.time()
        train(m, Xtr[:n_train], ytr[:n_train], epochs)
        acc = evaluate(m, Xte[:2000], yte[:2000])
        print(f"  {label:18s} 准确率={acc:.4f}  稀疏度={m.total_sparsity:.3f}  用时={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
