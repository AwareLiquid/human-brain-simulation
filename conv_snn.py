"""conv_snn.py — 卷积脉冲 SNN (CIFAR-10, surrogate gradient 直接训练)。

卷积层 + LIF 脉冲神经元 + 池化, 让稀疏脉冲真正对标 CNN。
"""
import argparse
import time
import pickle
import tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from awareliquid import surrogate_spike

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LIFConv(nn.Module):
    """卷积 → LIF 脉冲(T步) → 池化。"""

    def __init__(self, in_c, out_c, T, k=3, pool=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=1)
        self.pool = nn.MaxPool2d(pool)
        self.T = T
        self.threshold = 1.0
        self.decay = 0.9

    def forward(self, x):
        I = self.conv(x)                    # (B, C, H, W) 恒定电流
        v = torch.zeros_like(I)
        rate = torch.zeros_like(I)
        for _ in range(self.T):
            v = v * self.decay + I
            s = surrogate_spike(v, self.threshold)
            v = v * (1.0 - s)
            rate = rate + s
        return self.pool(rate / self.T)     # 池化 firing rate


class ConvSNN(nn.Module):
    def __init__(self, T=5, num_classes=10):
        super().__init__()
        self.l1 = LIFConv(3, 64, T)
        self.l2 = LIFConv(64, 128, T)
        self.l3 = LIFConv(128, 256, T)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)   # 32→16→8→4

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x), x.mean()


class DenseCNN(nn.Module):
    """稠密 CNN 基线 (ReLU, 同架构)。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.l3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.l3(self.l2(self.l1(x)))
        return self.fc(x.view(x.size(0), -1))


def load_cifar10():
    tar_path = 'data/cifar10-python.tar.gz'
    with tarfile.open(tar_path) as tf:
        def load_batch(name):
            f = tf.extractfile(f'cifar-10-batches-py/{name}')
            d = pickle.load(f, encoding='bytes')
            return d[b'data'], np.array(d[b'labels'])
        Xs, ys = [], []
        for i in range(1, 6):
            X, y = load_batch(f'data_batch_{i}')
            Xs.append(X); ys.append(y)
        Xtr = np.concatenate(Xs)
        ytr = np.concatenate(ys)
        Xte, yte = load_batch('test_batch')
    # (N, 3072) -> (N, 3, 32, 32), 归一化
    Xtr = Xtr.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    Xte = Xte.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64)


def train(model, X, y, epochs, batch, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
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
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {ep+1}/{epochs}: loss={total/n:.4f}", flush=True)


def evaluate(model, X, y):
    model.eval()
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    with torch.no_grad():
        out = model(Xt)
        if isinstance(out, tuple):
            out = out[0]
        return float((out.argmax(1) == yt).float().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--with-cnn', action='store_true')
    args = p.parse_args()

    print(f"=== 卷积 SNN 训练 (CIFAR-10) ===")
    print(f"device={device} T={args.T} epochs={args.epochs}")
    Xtr, ytr, Xte, yte = load_cifar10()
    print(f"数据: train={len(Xtr)} test={len(Xte)} shape={Xtr.shape}")

    model = ConvSNN(T=args.T).to(device)
    t0 = time.time()
    train(model, Xtr, ytr, args.epochs, args.batch, args.lr)
    acc = evaluate(model, Xte, yte)
    print(f"\n卷积 SNN: 准确率={acc:.4f}  用时={time.time()-t0:.0f}s", flush=True)

    if args.with_cnn:
        cnn = DenseCNN().to(device)
        t0 = time.time()
        train(cnn, Xtr, ytr, args.epochs, args.batch, args.lr)
        cnn_acc = evaluate(cnn, Xte, yte)
        print(f"稠密 CNN: 准确率={cnn_acc:.4f}  用时={time.time()-t0:.0f}s", flush=True)
        print(f"\n差距: {cnn_acc - acc:.4f} 点")


if __name__ == '__main__':
    main()
