"""awareliquid 包 — 类脑稀疏脉冲神经网络库。

从真实果蝇连接组提取统计规律, 构建可训练的稀疏 SNN。
"""
from .masks import longtail_mask, smallworld_mask
from .neurons import surrogate_spike
from .layers import SparseLIF
from .models import SparseSNN
from .recurrent import RecurrentSNN

__all__ = [
    "longtail_mask", "smallworld_mask",
    "surrogate_spike", "SparseLIF", "SparseSNN", "RecurrentSNN",
]
