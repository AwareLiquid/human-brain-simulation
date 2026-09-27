"""resnet_snn.py — ResNet-18 全深度脉冲网络, 挑战 90%+。

v2 改动（相对 84.76% 版本）：
- 4 层 × 2 残差块（64/128/256/512 通道），对齐标准 ResNet-18 深度
- weight decay 5e-4 + label smoothing 0.1
- 120 epochs（原 60）
- evaluate 分块（512 通道整集前向会 OOM）
- T=5 保持（VGG 实测 T=5 > T=10）
"""
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


def make_stage(in_c, out_c, T, stride, n_blocks=2):
    blocks = [ResBlock(in_c, out_c, T, stride=stride)]
    for _ in range(n_blocks - 1):
        blocks.append(ResBlock(out_c, out_c, T))
    return nn.Sequential(*blocks)


class ResNetSNN(nn.Module):
    """ResNet-18 全深度版 (4 层 × 2 块, 64/128/256/512)。"""

    def __init__(self, T=5, num_classes=10):
        super().__init__()
        self.head = LIFConv(3, 64, T)
        self.stage1 = make_stage(64, 64, T, stride=1)
        self.stage2 = make_stage(64, 128, T, stride=2)
        self.stage3 = make_stage(128, 256, T, stride=2)
        self.stage4 = make_stage(256, 512, T, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.head(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
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
    print("=== ResNet-18 全深度 SNN (T=5, 120 epochs, 增强+Cosine+wd5e-4) ===")
    model = ResNetSNN(T=5).to(device)
    train(model, Xtr, ytr, 120, 128, 1e-3)
    acc = evaluate(model, Xte, yte)
    print(f"ResNet SNN: {acc:.4f}")


if __name__ == '__main__':
    main()
