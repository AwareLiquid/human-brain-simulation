"""resnet_snn.py — ResNet 风格脉冲网络 (残差连接), 挑战 90%+。"""
import pickle
import tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from awareliquid import surrogate_spike

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LIFConv(nn.Module):
    def __init__(self, in_c, out_c, T, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.T = T
        self.threshold = 1.0
        self.decay = 0.9

    def forward(self, x):
        I = self.conv(x)
        v = torch.zeros_like(I)
        rate = torch.zeros_like(I)
        for _ in range(self.T):
            v = v * self.decay + I
            s = surrogate_spike(v, self.threshold)
            v = v * (1.0 - s)
            rate = rate + s
        return self.bn(rate / self.T)


class ResBlock(nn.Module):
    """残差块: Conv+LIF → Conv+LIF, 加残差 (firing rate 残差)。"""

    def __init__(self, in_c, out_c, T, stride=1):
        super().__init__()
        self.c1 = LIFConv(in_c, out_c, T, stride=stride)
        self.c2 = LIFConv(out_c, out_c, T)
        self.shortcut = None
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = self.c2(self.c1(x))
        sc = x if self.shortcut is None else self.shortcut(x)
        return out + sc


class ResNetSNN(nn.Module):
    """ResNet-18 风格 (CIFAR-10 简化版)。"""

    def __init__(self, T=5, num_classes=10):
        super().__init__()
        self.head = LIFConv(3, 96, T)
        self.b1 = ResBlock(96, 96, T)
        self.b2 = ResBlock(96, 192, T, stride=2)
        self.b3 = ResBlock(192, 384, T, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.head(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1)), x.mean()


def augment(x):
    B = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4), mode='reflect')
    top = torch.randint(0, 9, (1,)).item()
    left = torch.randint(0, 9, (1,)).item()
    out = padded[:, :, top:top+32, left:left+32]
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = torch.flip(out[flip], dims=[3])
    return out


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
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            out = model(augment(Xt[idx]))
            if isinstance(out, tuple): out = out[0]
            loss = F.cross_entropy(out, yt[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()*len(idx)
        sched.step()
        print(f"    epoch {ep+1}/{epochs}: loss={total/n:.4f}", flush=True)


def evaluate(model, X, y, batch=256):
    model.eval()
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    correct, n = 0, len(Xt)
    with torch.no_grad():
        for i in range(0, n, batch):
            out = model(Xt[i:i+batch])
            if isinstance(out, tuple): out = out[0]
            correct += (out.argmax(1) == yt[i:i+batch]).sum().item()
    return correct / n


def main():
    Xtr, ytr, Xte, yte = load_cifar10()
    print("=== ResNet SNN Wide (T=5, 70 epochs, 96/192/384, 增强+Cosine+wd5e-4) ===")
    model = ResNetSNN(T=5).to(device)
    train(model, Xtr, ytr, 70, 256, 1e-3)
    acc = evaluate(model, Xte, yte)
    print(f"ResNet SNN: {acc:.4f}")


if __name__ == '__main__':
    main()
