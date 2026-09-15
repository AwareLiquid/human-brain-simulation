"""conv_conversion_v2.py — 改进转换法: 99.9分位数归一化 + AvgPool。

探索模式假设 A': 分位数归一化 + AvgPool 能否把转换法推到 72%+。
"""
import pickle
import tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DenseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)   # 用 AvgPool (脉冲域更友好)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv3(x)), 2)
        return self.fc(x.view(x.size(0), -1))


class SpikeCNN(nn.Module):
    def __init__(self, cnn, scales, T):
        super().__init__()
        self.T = T
        self.thr = 1.0
        s1, s2, s3 = scales   # 99.9 分位数
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1.weight.data = cnn.conv1.weight.data / s1
        self.conv1.bias.data = cnn.conv1.bias.data / s1
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2.weight.data = cnn.conv2.weight.data * s1 / s2
        self.conv2.bias.data = cnn.conv2.bias.data * s1 / s2
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3.weight.data = cnn.conv3.weight.data * s2 / s3
        self.conv3.bias.data = cnn.conv3.bias.data * s2 / s3
        self.fc = nn.Linear(256 * 4 * 4, 10)
        self.fc.weight.data = cnn.fc.weight.data * s3
        self.fc.bias.data = cnn.fc.bias.data

    def _lif(self, x):
        v = torch.zeros_like(x)
        spikes = torch.zeros_like(x)
        for _ in range(self.T):
            v = v + x
            s = (v >= self.thr).float()
            v = v - s * self.thr
            spikes = spikes + s
        return spikes / self.T

    def forward(self, x):
        x = self._lif(self.conv1(x)); x = F.avg_pool2d(x, 2)
        x = self._lif(self.conv2(x)); x = F.avg_pool2d(x, 2)
        x = self._lif(self.conv3(x)); x = F.avg_pool2d(x, 2)
        return self.fc(x.view(x.size(0), -1))


def load_cifar10():
    with tarfile.open('data/cifar10-python.tar.gz') as tf:
        def lb(n):
            d = pickle.load(tf.extractfile(f'cifar-10-batches-py/{n}'), encoding='bytes')
            return d[b'data'], np.array(d[b'labels'])
        Xs, ys = [], []
        for i in range(1, 6):
            X, y = lb(f'data_batch_{i}'); Xs.append(X); ys.append(y)
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        Xte, yte = lb('test_batch')
    return (Xtr.reshape(-1, 3, 32, 32).astype(np.float32)/255.0, ytr.astype(np.int64),
            Xte.reshape(-1, 3, 32, 32).astype(np.float32)/255.0, yte.astype(np.int64))


def train(model, X, y, epochs, batch, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()*len(idx)
        print(f"    epoch {ep+1}/{epochs}: loss={total/n:.4f}", flush=True)


def evaluate(model, X, y):
    model.eval()
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    with torch.no_grad():
        return float((model(Xt).argmax(1) == yt).float().mean().item())


def main():
    Xtr, ytr, Xte, yte = load_cifar10()
    print("=== 1. 训练稠密 CNN (AvgPool) ===")
    cnn = DenseCNN().to(device)
    train(cnn, Xtr, ytr, 15, 128, 1e-3)
    acc_cnn = evaluate(cnn, Xte, yte)
    print(f"稠密 CNN: {acc_cnn:.4f}")

    cnn.eval()
    with torch.no_grad():
        x = torch.tensor(Xtr[:2000], device=device)
        h1 = F.relu(cnn.conv1(x)); s1 = float(np.percentile(h1.detach().cpu().numpy(), 99.9))
        x1 = F.avg_pool2d(h1, 2)
        h2 = F.relu(cnn.conv2(x1)); s2 = float(np.percentile(h2.detach().cpu().numpy(), 99.9))
        x2 = F.avg_pool2d(h2, 2)
        h3 = F.relu(cnn.conv3(x2)); s3 = float(np.percentile(h3.detach().cpu().numpy(), 99.9))
    print(f"99.9分位数: s1={s1:.2f} s2={s2:.2f} s3={s3:.2f}")

    print("=== 2. 转换脉冲卷积 (分位数归一化 + AvgPool) ===")
    for T in [10, 20, 50, 100]:
        snn = SpikeCNN(cnn, (s1, s2, s3), T).to(device)
        acc = evaluate(snn, Xte, yte)
        print(f"T={T:3d}: {acc:.4f}  (稠密 CNN {acc_cnn:.4f})")
    print(f"\n证伪判据: ≥72%?")


if __name__ == '__main__':
    main()
