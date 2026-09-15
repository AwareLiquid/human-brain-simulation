"""train_large.py — 大规模稀疏 SNN 训练 (GPU / A100)。

支持 MNIST / Fashion-MNIST / CIFAR-10, 可配置多层稀疏架构 + 能耗对比。

用法 (在 A100 上):
  pip install torch torchvision numpy scipy pyarrow safetensors
  python train_large.py --dataset cifar10 --hidden 2000 --layers 3 --density 0.05 --epochs 20

数据: MNIST/Fashion 用 data/ 下已有的 numpy (download_data.py 下载);
      CIFAR-10 用 torchvision 自动下载。
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from awareliquid import SparseSNN, longtail_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(dataset):
    """返回 (Xtr, ytr, Xte, yte, in_dim, num_classes)。"""
    if dataset == 'cifar10':
        import os, pickle, tarfile, urllib.request
        tar_path = 'data/cifar10-python.tar.gz'
        if not os.path.exists(tar_path):
            print("  下载 CIFAR-10 (~170MB)...")
            urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', tar_path)
        def load_batch(tf, name):
            f = tf.extractfile(f'cifar-10-batches-py/{name}')
            d = pickle.load(f, encoding='bytes')
            return d[b'data'], np.array(d[b'labels'])
        with tarfile.open(tar_path) as tf:
            Xs, ys = [], []
            for i in range(1, 6):
                X, y = load_batch(tf, f'data_batch_{i}')
                Xs.append(X); ys.append(y)
            Xtr = np.concatenate(Xs).astype(np.float32) / 255.0
            ytr = np.concatenate(ys).astype(np.int64)
            Xte, yte = load_batch(tf, 'test_batch')
            Xte = Xte.astype(np.float32) / 255.0
            yte = yte.astype(np.int64)
        return Xtr, ytr, Xte, yte, 3 * 32 * 32, 10

    # mnist / fashion: numpy
    if dataset == 'fashion':
        import gzip
        def imgs(p):
            with gzip.open(p, 'rb') as f:
                f.read(4); n = int.from_bytes(f.read(4), 'big')
                r = int.from_bytes(f.read(4), 'big'); c = int.from_bytes(f.read(4), 'big')
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)
        def labels(p):
            with gzip.open(p, 'rb') as f:
                f.read(4); n = int.from_bytes(f.read(4), 'big')
                return np.frombuffer(f.read(), dtype=np.uint8)
        d = 'data/fashion/'
        Xtr = imgs(d + 'train-images-idx3-ubyte.gz').astype(np.float32) / 255.0
        ytr = labels(d + 'train-labels-idx1-ubyte.gz').astype(np.int64)
        Xte = imgs(d + 't10k-images-idx3-ubyte.gz').astype(np.float32) / 255.0
        yte = labels(d + 't10k-labels-idx1-ubyte.gz').astype(np.int64)
        return Xtr, ytr, Xte, yte, 784, 10
    else:  # mnist
        d = np.load('data/mnist.npz')
        Xtr = d['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
        ytr = d['y_train'].astype(np.int64)
        Xte = d['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
        yte = d['y_test'].astype(np.int64)
        return Xtr, ytr, Xte, yte, 784, 10


def build_model(in_dim, num_classes, hidden, layers, density, T=15):
    dims = [in_dim] + [hidden] * layers
    masks, signs = [], []
    for i in range(len(dims) - 1):
        m, s = longtail_mask(dims[i + 1], dims[i], density)
        masks.append(m.to(device))
        signs.append(s.to(device))
    return SparseSNN(dims, masks, signs, num_classes, T=T, inter_gain=5.0).to(device)


class DenseMLP(nn.Module):
    def __init__(self, in_dim, hidden, layers, num_classes):
        super().__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_dim, hidden))
        for _ in range(layers - 1):
            self.fcs.append(nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x))
        return self.out(x)


def train(model, Xtr, ytr, epochs, batch, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = model(Xt[idx])
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {ep+1}/{epochs}: loss={total/n:.4f}")


def evaluate(model, Xte, yte):
    model.eval()
    Xt = torch.tensor(Xte, device=device)
    yt = torch.tensor(yte, device=device)
    with torch.no_grad():
        out = model(Xt)
        if isinstance(out, tuple):
            out = out[0]
        return float((out.argmax(1) == yt).float().mean().item())


def energy_compare(model, in_dim, hidden, layers, num_classes):
    """稀疏 SNN vs 稠密 MLP 能耗 (45nm 模型, MAC 3.7pJ / 加法 0.9pJ)。"""
    dense_flops = 2 * (in_dim * hidden + (layers - 1) * hidden * hidden + hidden * num_classes)
    conn = sum(int(l.mask.sum()) for l in model.layers) + hidden * num_classes
    fr = 0.1
    sops = conn * fr * model.T
    return dense_flops * 3.7 / (sops * 0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10', choices=['mnist', 'fashion', 'cifar10'])
    p.add_argument('--hidden', type=int, default=2000)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--density', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--T', type=int, default=15)
    p.add_argument('--with-mlp', action='store_true', help='同时训练稠密 MLP 基线')
    args = p.parse_args()

    print(f"=== 大规模稀疏 SNN 训练 ===")
    print(f"device={device}  dataset={args.dataset}  hidden={args.hidden}  "
          f"layers={args.layers}  density={args.density}  epochs={args.epochs}")
    if not torch.cuda.is_available():
        print("⚠️  未检测到 CUDA! 请确认安装了 CUDA 版 torch (pip install torch --index-url https://download.pytorch.org/whl/cu128)")

    Xtr, ytr, Xte, yte, in_dim, num_classes = load_data(args.dataset)
    print(f"数据: train={len(Xtr)}  test={len(Xte)}  in_dim={in_dim}")

    # 稀疏 SNN
    model = build_model(in_dim, num_classes, args.hidden, args.layers, args.density, args.T)
    print(f"模型: 稀疏度={model.total_sparsity:.3f} ({model.total_sparsity*100:.1f}% 稀疏)")
    t0 = time.time()
    train(model, Xtr, ytr, args.epochs, args.batch, args.lr)
    acc = evaluate(model, Xte, yte)
    saving = energy_compare(model, in_dim, args.hidden, args.layers, num_classes)
    print(f"\n稀疏 SNN: 准确率={acc:.4f}  能耗省 {saving:.0f}x  用时={time.time()-t0:.0f}s")

    # 稠密 MLP 基线 (可选)
    if args.with_mlp:
        mlp = DenseMLP(in_dim, args.hidden, args.layers, num_classes).to(device)
        t0 = time.time()
        train(mlp, Xtr, ytr, args.epochs, args.batch, args.lr)
        mlp_acc = evaluate(mlp, Xte, yte)
        print(f"稠密 MLP: 准确率={mlp_acc:.4f}  用时={time.time()-t0:.0f}s")
        print(f"\n差距: {mlp_acc - acc:.4f} 点 | 能耗省 {saving:.0f}x")


if __name__ == '__main__':
    main()
