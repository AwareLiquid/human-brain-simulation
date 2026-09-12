"""脉冲神经元 — surrogate gradient 的 LIF (漏积分放电)。"""
import torch


def surrogate_spike(v, threshold=1.0):
    """STE surrogate: 前向硬阈值 spike, 反向 sigmoid 梯度。

    前向: spike = (v > threshold)  (0/1, 无梯度)
    反向: 梯度 = sigmoid(v - threshold) 的导数 (平滑近似)
    """
    spike = (v > threshold).float()
    return spike + torch.sigmoid(v - threshold) - torch.sigmoid(v - threshold).detach()


def lif_step(v, I, decay, threshold):
    """单步 LIF: v <- decay*v + I; spike <- (v>thr); v <- reset。返回 (新 v, spike)。"""
    v = v * decay + I
    s = surrogate_spike(v, threshold)
    v = v * (1.0 - s)          # 硬 reset
    return v, s
