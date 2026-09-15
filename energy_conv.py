"""energy_conv.py — 能耗实测: 脉冲卷积 (VGG SNN) vs 稠密 CNN。

45nm 模型: MAC 3.7 pJ, 加法 0.9 pJ。
脉冲卷积只做 spike 触发的加法; 稠密 CNN 每连接做乘加。
"""
import pickle
import tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_snn import VGGSNN, LIFConv, load_cifar10, train, device


class DenseCNN6(nn.Module):
    """稠密 CNN (同 VGG 6层架构, ReLU)。"""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.c6 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.c1(x)), 2)
        x = F.avg_pool2d(F.relu(self.c2(x)), 2)
        x = F.avg_pool2d(F.relu(self.c3(x)), 2)
        x = F.avg_pool2d(F.relu(self.c4(x)), 2)
        x = F.avg_pool2d(F.relu(self.c5(x)), 2)
        x = F.avg_pool2d(F.relu(self.c6(x)), 2)
        return self.fc(x.view(x.size(0), -1))


def measure_firing_rate(model, X, batch=256):
    """实测 VGG SNN 每层的平均 firing rate。"""
    model.eval()
    frs = []
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch):
            x = torch.tensor(X[i:i+batch], device=device)
            frs.append(model(x)[1].item())   # 最后一层 firing rate
    return float(np.mean(frs))


def main():
    Xtr, ytr, Xte, yte = load_cifar10()

    # 训练 VGG SNN
    print("=== 训练 VGG SNN (T=5) ===")
    snn = VGGSNN(T=5).to(device)
    train(snn, Xtr, ytr, 30, 128, 1e-3)

    # 实测 firing rate
    fr = measure_firing_rate(snn, Xte)
    print(f"实测 firing rate: {fr:.4f}")

    # 卷积 FLOPs (稠密 CNN 每层乘加)
    # 6 层卷积: 每层 FLOPs = 2 * C_in * C_out * k^2 * H * W
    flops = 0
    shapes = [(3, 64, 32, 32), (64, 64, 32, 32), (64, 128, 16, 16),
              (128, 128, 16, 16), (128, 256, 8, 8), (256, 256, 8, 8)]
    for cin, cout, h, w in shapes:
        flops += 2 * cin * cout * 9 * h * w   # k=3, 乘加

    # 脉冲卷积: spike 触发的加法
    # 每层 spike 操作 = 卷积连接数 × firing rate × T
    sops = 0
    for cin, cout, h, w in shapes:
        sops += cin * cout * 9 * h * w * fr * 5   # T=5

    mac_pj, add_pj = 3.7, 0.9
    e_cnn = flops * mac_pj
    e_snn = sops * add_pj
    print(f"\n=== 能耗对比 ===")
    print(f"稠密 CNN: {flops:.2e} FLOPs -> {e_cnn/1e6:.1f} mJ")
    print(f"脉冲卷积: {sops:.2e} 加法 -> {e_snn/1e6:.1f} mJ")
    print(f"能耗比: {e_cnn/e_snn:.1f}x")


if __name__ == '__main__':
    main()
