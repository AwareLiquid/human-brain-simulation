"""full_benchmark.py — 两层 SNN 全量训练 (最终准确率验证)。"""
import time
from benchmark import build_model, train, evaluate, load_mnist


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    print("=== 两层 SNN (784→800→800→10) 全量训练 (60000 样本, 12 epochs) ===")
    m = build_model([784, 800, 800], 0.05, inter_gain=5.0)
    t0 = time.time()
    train(m, Xtr, ytr, epochs=12)
    acc = evaluate(m, Xte, yte)
    print(f"\n准确率={acc:.4f}  总稀疏度={m.total_sparsity:.3f}  用时={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
