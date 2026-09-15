"""vgg_snn.py — VGG 风格深度卷积 SNN (6层卷积, 挑战 80%+)。"""
import pickle
import tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from awareliquid import surrogate_spike

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LIFConv(nn.Module):
    def __init__(self, in_c, out_c, T, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.AvgPool2d(2) if pool else None
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
        rate = self.bn(rate / self.T)
        return self.pool(rate) if self.pool else rate


class VGGSNN(nn.Module):
    """VGG-9 风格: 64-64-P-128-128-P-256-256-P → FC。"""

    def __init__(self, T=5, num_classes=10):
        super().__init__()
        self.l1 = LIFConv(3, 64, T)
        self.l2 = LIFConv(64, 64, T, pool=True)
        self.l3 = LIFConv(64, 128, T)
        self.l4 = LIFConv(128, 128, T, pool=True)
        self.l5 = LIFConv(128, 256, T)
        self.l6 = LIFConv(256, 256, T, pool=True)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.l1(x); x = self.l2(x)
        x = self.l3(x); x = self.l4(x)
        x = self.l5(x); x = self.l6(x)
        return self.fc(x.view(x.size(0), -1)), x.mean()


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
            out = model(Xt[idx])
            if isinstance(out, tuple): out = out[0]
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()*len(idx)
        print(f"    epoch {ep+1}/{epochs}: loss={total/n:.4f}", flush=True)


def evaluate(model, X, y):
    model.eval()
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    with torch.no_grad():
        out = model(Xt)
        if isinstance(out, tuple): out = out[0]
        return float((out.argmax(1) == yt).float().mean().item())


def main():
    Xtr, ytr, Xte, yte = load_cifar10()
    for T in [5, 10]:
        print(f"=== VGG 深度卷积 SNN (T={T}, 30 epochs) ===")
        model = VGGSNN(T=T).to(device)
        train(model, Xtr, ytr, 30, 128, 1e-3)
        acc = evaluate(model, Xte, yte)
        print(f"VGG SNN T={T}: {acc:.4f}")


if __name__ == '__main__':
    main()
