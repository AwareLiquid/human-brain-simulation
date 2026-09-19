# -*- coding: utf-8 -*-
"""train_and_save.py — 训练稀疏 SNN (MNIST + Fashion-MNIST) 并保存权重。

在 Digital Human Brain 仓库根目录运行（数据路径为相对路径）。
用法:
  python train_and_save.py --smoke    # 冒烟测试 (1 epoch, 2000 样本)
  python train_and_save.py            # 完整训练 (12 epochs, 全量)
"""
import sys, os, time, json, argparse
import numpy as np
import torch
sys.stdout.reconfigure(encoding='utf-8')

from sparse_snn import (load_mnist, SparseSNN, longtail_mask_ei, train_model, evaluate, device)

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()

os.makedirs('checkpoints', exist_ok=True)
IN_DIM, HID_DIM, OUT_DIM = 784, 800, 10
EPOCHS = 12
DENSITY = 0.05


def load_fashion():
    import gzip
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


def run(name, Xtr, ytr, Xte, yte, epochs):
    print('=' * 60)
    print(f'### {name} | device={device} | epochs={epochs} | train={len(Xtr)}')
    mask, sign = longtail_mask_ei(HID_DIM, IN_DIM, DENSITY)
    # 注: 原 sparse_snn.py 的 w_hid 初始化要求 sign 在 CPU (randn 在 CPU);
    # 传入 CPU 张量, 由 .to(device) 统一搬运 (mask 是 buffer 会一起搬)。
    snn = SparseSNN(IN_DIM, HID_DIM, OUT_DIM, mask, sign).to(device)
    t0 = time.time()
    train_model(snn, Xtr, ytr, epochs)
    acc, fr = evaluate(snn, Xte, yte)
    dt = time.time() - t0
    print(f'  准确率={acc:.4f}  firing_rate={fr:.4f}  用时={dt:.0f}s')

    ckpt = {
        'state_dict': {k: v.cpu() for k, v in snn.state_dict().items()},
        'sign': sign.cpu(),
        'config': {
            'in_dim': IN_DIM, 'hid_dim': HID_DIM, 'out_dim': OUT_DIM,
            'T': 15, 'decay': 0.9, 'threshold': 1.0,
            'density': DENSITY, 'e_ratio': 0.6, 'mask_seed': 0,
            'dataset': name, 'epochs': epochs,
        },
        'metrics': {
            'test_acc': acc, 'firing_rate': fr,
            'train_seconds': round(dt, 1),
            'train_samples': int(len(Xtr)),
            'device': str(device),
        },
    }
    out = f'checkpoints/sparse_snn_{name}.pt'
    torch.save(ckpt, out)
    print(f'  saved: {out} ({os.path.getsize(out)/1e6:.2f} MB)')
    return ckpt['metrics']


if args.smoke:
    print('*** SMOKE MODE ***')
    Xtr, ytr, Xte, yte = load_mnist()
    m = run('mnist_smoke', Xtr[:2000], ytr[:2000], Xte[:500], yte[:500], 1)
    print(json.dumps(m, indent=1))
    sys.exit(0)

results = {}
Xtr, ytr, Xte, yte = load_mnist()
results['mnist'] = run('mnist', Xtr, ytr, Xte, yte, EPOCHS)

Xtr, ytr, Xte, yte = load_fashion()
results['fashion'] = run('fashion', Xtr, ytr, Xte, yte, EPOCHS)

print()
print('=== 汇总 ===')
print(json.dumps(results, indent=1))
with open('checkpoints/train_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=1)
print('done.')
