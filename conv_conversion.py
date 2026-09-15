"""conv_conversion.py — 卷积转换法 (稠密 CNN → 脉冲卷积, firing rate 编码)。

探索模式假设 A: 转换法能否把稠密 CNN 的准确率几乎无损地转成脉冲卷积。
证伪判据: 转换后准确率 ≥ 72% (差距 <3 点), 否则脉冲量化是卷积的硬瓶颈。
"""
import argparse
import time
import pickle
import tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DenseCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return self.fc(x.view(x.size(0), -1))


class SpikeCNN(nn.Module):
    """由 DenseCNN 转换的脉冲卷积 (IF 神经元 + firing rate 编码)。"""

    def __init__(self, cnn, max1, max2, max3, T):
        super().__init__()
        self.T = T
        self.thr = 1.0
        # 权重归一化: 层 i 除以 max_i, 下一层乘以 max_i (补偿 scale)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1.weight.data = cnn.conv1.weight.data / max1
        self.conv1.bias.data = cnn.conv1.bias.data / max1
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2.weight.data = cnn.conv2.weight.data * max1 / max2
        self.conv2.bias.data = cnn.conv2.bias.data * max1 / max2
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3.weight.data = cnn.conv3.weight.data * max2 / max3
        self.conv3.bias.data = cnn.conv3.bias.data * max2 / max3
        self.fc = nn.Linear(256 * 4 * 4, 10)
        self.fc.weight.data = cnn.fc.weight.data * max3
        self.fc.bias.data = cnn.fc.bias.data

    def _lif(self, x):
        """IF 神经元: firing rate ≈ ReLU(输入电流)。"""
        v = torch.zeros_like(x)
        spikes = torch.zeros_like(x)
        for _ in range(self.T):
            v = v + x
            s = (v >= self.thr).float()
            v = v - s * self.thr
            spikes = spikes + s
        return spikes / self.T

    def forward(self, x):
        x = self._lif(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self._lif(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self._lif(self.conv3(x))
        x = F.max_pool2d(x, 2)
        return self.fc(x.view(x.size(0), -1))


def load_cifar10():
    with tarfile.open('data/cifar10-python.tar.gz') as tf:
        def load_batch(name):
            d = pickle.load(tf.extractfile(f'cifar-10-batches-py/{name}'), encoding='bytes')
            return d[b'data'], np.array(d[b'labels'])
        Xs, ys = [], []
        for i in range(1, 6):
            X, y = load_batch(f'data_batch_{i}')
            Xs.append(X); ys.append(y)
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        Xte, yte = load_batch('test_batch')
    return (Xtr.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0, ytr.astype(np.int64),
            Xte.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0, yte.astype(np.int64))


def train(model, X, y, epochs, batch, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = model(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item() * len(idx)
        print(f"    epoch {ep+1}/{epochs}: loss={total/n:.4f}", flush=True)


def evaluate(model, X, y):
    model.eval()
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    with torch.no_grad():
        out = model(Xt)
        return float((out.argmax(1) == yt).float().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    Xtr, ytr, Xte, yte = load_cifar10()
    print(f"device={device} 数据: train={len(Xtr)}")

    # 1. 训练稠密 CNN
    print("=== 1. 训练稠密 CNN ===")
    cnn = DenseCNN().to(device)
    train(cnn, Xtr, ytr, args.epochs, args.batch, args.lr)
    acc_cnn = evaluate(cnn, Xte, yte)
    print(f"稠密 CNN: {acc_cnn:.4f}")

    # 2. 统计每层最大激活 (归一化用)
    cnn.eval()
    with torch.no_grad():
        x = torch.tensor(Xtr[:2000], device=device)
        h1 = F.relu(cnn.conv1(x)); max1 = float(h1.max())
        x1 = F.max_pool2d(h1, 2)
        h2 = F.relu(cnn.conv2(x1)); max2 = float(h2.max())
        x2 = F.max_pool2d(h2, 2)
        h3 = F.relu(cnn.conv3(x2)); max3 = float(h3.max())
    print(f"最大激活: max1={max1:.2f} max2={max2:.2f} max3={max3:.2f}")

    # 3. 转换脉冲卷积, 不同 T
    print("=== 2. 转换脉冲卷积 (firing rate 编码) ===")
    for T in [5, 10, 20, 50]:
        snn = SpikeCNN(cnn, max1, max2, max3, T).to(device)
        acc_snn = evaluate(snn, Xte, yte)
        print(f"T={T:3d}: 脉冲卷积 {acc_snn:.4f}  (稠密 CNN {acc_cnn:.4f})")

    # 证伪判据
    print(f"\n证伪判据: 转换后 ≥ 72%? (差距 <3 点)")


if __name__ == '__main__':
    main()
