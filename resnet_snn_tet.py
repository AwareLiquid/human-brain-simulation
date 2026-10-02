"""resnet_snn_tet.py — TET (Temporal Efficient Training, Deng et al. NeurIPS
2022) applied to the ResNet SNN: per-timestep losses with exponential
temporal weights, the "remaining lever" for 90%+ after the 7-round series
plateaued at 84.89% (EXPERIMENTS.md).

TET change vs resnet_snn.py: instead of averaging spikes into a firing
rate, every LIF layer keeps its membrane across timesteps and emits an
output at EVERY step; the loss is the sum over steps of
lambda(t) * CE(logits_t, y) with lambda(t) = 2^(t-T). Eval reads the last
step (standard TET).

Run:  python resnet_snn_tet.py
"""

import pickle
import tarfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from awareliquid import surrogate_spike

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LIFConvTET(nn.Module):
    """Conv+LIF with a PERSISTENT membrane across timesteps; one step per call."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.threshold = 1.0
        self.decay = 0.9
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        I = self.conv(x)
        if self.v is None:
            self.v = torch.zeros_like(I)
        self.v = self.v * self.decay + I
        s = surrogate_spike(self.v, self.threshold)
        self.v = self.v * (1.0 - s)
        return self.bn(s)            # this step's output (post-BN spike)


class ResBlockTET(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.c1 = LIFConvTET(in_c, out_c, stride=stride)
        self.c2 = LIFConvTET(out_c, out_c)
        self.shortcut = None
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c))

    def reset(self):
        self.c1.reset()
        self.c2.reset()

    def forward(self, x):
        out = self.c2(self.c1(x))
        sc = x if self.shortcut is None else self.shortcut(x)
        return out + sc


class ResNetSNNTET(nn.Module):
    def __init__(self, T=5, num_classes=10):
        super().__init__()
        self.T = T
        self.head = LIFConvTET(3, 64)
        self.b1 = ResBlockTET(64, 64)
        self.b2 = ResBlockTET(64, 128, stride=2)
        self.b3 = ResBlockTET(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def reset(self):
        self.head.reset()
        self.b1.reset()
        self.b2.reset()
        self.b3.reset()

    def forward(self, x):
        """x: (B,3,32,32) -> logits (B, T, C) — one readout per timestep."""
        self.reset()
        outs = []
        for _ in range(self.T):
            h = self.b3(self.b2(self.b1(self.head(x))))
            h = self.pool(h).view(h.size(0), -1)
            outs.append(self.fc(h))
        return torch.stack(outs, dim=1)


def tet_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum over timesteps of lambda(t) * CE, lambda(t) = 2^(t-T)."""
    T = logits.shape[1]
    lambdas = torch.tensor([2.0 ** (t - T) for t in range(T)],
                           device=logits.device)     # 2^(1-T) ... 1
    per_t = torch.stack(
        [F.cross_entropy(logits[:, t], y) for t in range(T)])
    return (lambdas * per_t).sum() / lambdas.sum()


def augment(x):
    B = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4), mode='reflect')
    top = torch.randint(0, 9, (1,)).item()
    left = torch.randint(0, 9, (1,)).item()
    out = padded[:, :, top:top + 32, left:left + 32]
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = torch.flip(out[flip], dims=[3])
    return out


def cutout(x, hole=16):
    B = x.shape[0]
    y0 = torch.randint(0, 32 - hole + 1, (B,), device=x.device)
    x0 = torch.randint(0, 32 - hole + 1, (B,), device=x.device)
    for b in range(B):
        x[b, :, y0[b]:y0[b] + hole, x0[b]:x0[b] + hole] = 0.0
    return x


def load_cifar10():
    with tarfile.open('data/cifar10-python.tar.gz') as tf:
        def lb(n):
            d = pickle.load(tf.extractfile(f'cifar-10-batches-py/{n}'),
                            encoding='bytes')
            return d[b'data'], np.array(d[b'labels'])
        Xs, ys = [], []
        for i in range(1, 6):
            X, y = lb(f'data_batch_{i}')
            Xs.append(X)
            ys.append(y)
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        Xte, yte = lb('test_batch')
    return (Xtr.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0,
            ytr.astype(np.int64),
            Xte.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0,
            yte.astype(np.int64))


def main():
    Xtr, ytr, Xte, yte = load_cifar10()
    print("=== ResNet SNN TET (T=5, 200 epochs, 增强+Cutout+Cosine) ===")
    model = ResNetSNNTET(T=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    n = len(Xt)

    for ep in range(200):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb = cutout(augment(Xt[idx]))
            logits = model(xb)                 # (B, T, C)
            loss = tet_loss(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        print(f"    epoch {ep + 1}/200: loss={total / n:.4f}", flush=True)

    # eval: last-timestep logits
    model.eval()
    Xe = torch.tensor(Xte, device=device)
    ye = torch.tensor(yte, device=device)
    correct, m = 0, len(Xe)
    with torch.no_grad():
        for i in range(0, m, 256):
            logits = model(Xe[i:i + 256])
            correct += (logits[:, -1].argmax(1) == ye[i:i + 256]).sum().item()
    print(f"ResNet SNN TET: {correct / m:.4f}")


if __name__ == '__main__':
    main()
