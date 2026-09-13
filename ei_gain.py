"""ei_gain.py — E/I 增益回填验证: 抑制 1x vs 4x (H01 人脑规律)。

验证 H01 人脑发现的「抑制是兴奋 ~4 倍」能否提升可训练稀疏 SNN。
"""
import time
from awareliquid import SparseSNN, longtail_mask
from benchmark import train, evaluate, load_mnist


def build(dims, density, inh_gain):
    masks, signs = [], []
    for i in range(len(dims) - 1):
        m, s = longtail_mask(dims[i + 1], dims[i], density, inh_gain=inh_gain)
        masks.append(m)
        signs.append(s)
    return SparseSNN(dims, masks, signs, 10, T=15, inter_gain=5.0)


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    n_train, epochs = 20000, 8

    print("=== E/I 增益对比 (MNIST, 20000 样本, 8 epochs, 2层800) ===")
    for inh_gain in [1.0, 2.0, 4.0]:
        m = build([784, 800, 800], 0.05, inh_gain)
        t0 = time.time()
        train(m, Xtr[:n_train], ytr[:n_train], epochs)
        acc = evaluate(m, Xte[:2000], yte[:2000])
        print(f"  抑制增益 {inh_gain:.0f}x: 准确率={acc:.4f}  用时={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
