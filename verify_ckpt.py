# -*- coding: utf-8 -*-
"""verify_ckpt.py — 验证保存的 checkpoint 能加载并复现精度"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from sparse_snn import SparseSNN, load_mnist, evaluate, device

# MNIST
ckpt = torch.load('checkpoints/sparse_snn_mnist.pt', map_location='cpu', weights_only=False)
cfg = ckpt['config']
m = SparseSNN(cfg['in_dim'], cfg['hid_dim'], cfg['out_dim'],
              ckpt['state_dict']['mask'], ckpt['sign'],
              T=cfg['T'], decay=cfg['decay'], threshold=cfg['threshold'])
m.load_state_dict(ckpt['state_dict'])
m = m.to(device)
m.eval()
Xtr, ytr, Xte, yte = load_mnist()
acc, fr = evaluate(m, Xte, yte)
print('MNIST loaded: acc=%.4f fr=%.4f | saved: acc=%.4f fr=%.4f' % (
    acc, fr, ckpt['metrics']['test_acc'], ckpt['metrics']['firing_rate']))

# Fashion
ckpt2 = torch.load('checkpoints/sparse_snn_fashion.pt', map_location='cpu', weights_only=False)
cfg2 = ckpt2['config']
m2 = SparseSNN(cfg2['in_dim'], cfg2['hid_dim'], cfg2['out_dim'],
               ckpt2['state_dict']['mask'], ckpt2['sign'],
               T=cfg2['T'], decay=cfg2['decay'], threshold=cfg2['threshold'])
m2.load_state_dict(ckpt2['state_dict'])
m2 = m2.to(device)
m2.eval()

import gzip
def imgs(p):
    with gzip.open(p, 'rb') as f:
        f.read(4); n = int.from_bytes(f.read(4),'big'); r = int.from_bytes(f.read(4),'big'); c = int.from_bytes(f.read(4),'big')
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r*c)
def labels(p):
    with gzip.open(p, 'rb') as f:
        f.read(4); n = int.from_bytes(f.read(4),'big')
        return np.frombuffer(f.read(), dtype=np.uint8)
Xte2 = imgs('data/fashion/t10k-images-idx3-ubyte.gz').astype(np.float32)/255.0
yte2 = labels('data/fashion/t10k-labels-idx1-ubyte.gz').astype(np.int64)
acc2, fr2 = evaluate(m2, Xte2, yte2)
print('Fashion loaded: acc=%.4f fr=%.4f | saved: acc=%.4f fr=%.4f' % (
    acc2, fr2, ckpt2['metrics']['test_acc'], ckpt2['metrics']['firing_rate']))
print('LOAD TEST PASSED' if abs(acc - ckpt['metrics']['test_acc']) < 1e-6 and abs(acc2 - ckpt2['metrics']['test_acc']) < 1e-6 else 'MISMATCH!')
