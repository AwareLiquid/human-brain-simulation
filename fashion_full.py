"""fashion_full.py — Fashion-MNIST (更难任务) 全量 + 多层大模型。

把 fashion_snn.py 的单层 87% 往上推: 2 层 1500 隐 + 全量 60000 + 12 epochs。
"""
import time
import torch
from fashion_snn import load_fashion
from benchmark import build_model, train, evaluate


def main():
    Xtr, ytr, Xte, yte = load_fashion()

    print("=== Fashion-MNIST 全量 + 2层1500 (更难任务完整验证) ===")
    m = build_model([784, 1500, 1500], 0.05, inter_gain=5.0)
    t0 = time.time()
    train(m, Xtr, ytr, epochs=12)
    acc = evaluate(m, Xte, yte)
    print(f"\n准确率={acc:.4f}  总稀疏度={m.total_sparsity:.3f}  用时={time.time()-t0:.0f}s")

    # 能耗对比 (稠密 MLP 基线参考: 全量 Fashion 单层 800 约 87.6%)
    flops_mlp = 2 * (784 * 1500 + 1500 * 1500 + 1500 * 10)
    conn = sum(int(l.mask.sum()) for l in m.layers) + 1500 * 10
    fr = 0.08  # 典型 firing rate
    sops = conn * fr * 15
    print(f"能耗: 稠密MLP {flops_mlp:.1e} FLOPs vs 稀疏SNN {sops:.1e} 加法 -> 省 {flops_mlp*3.7/(sops*0.9):.0f}x")


if __name__ == '__main__':
    main()
