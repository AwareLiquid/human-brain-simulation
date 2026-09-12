"""energy.py — 实验E: 稀疏脉冲储备池 vs 稠密 ANN 的能耗对比。

SNN 社区标准能耗模型 (Horowitz 2014, 45nm CMOS):
  MAC (乘加) = 3.7 pJ, AC (加法) = 0.9 pJ
脉冲网络事件驱动「只做加法」, 稠密网络「每连接做乘加」。
"""
import numpy as np
import scipy.sparse as sp


def main():
    # 参数来自已跑实验
    N = 3000
    T = 20
    fr = 0.096                      # 实测 firing rate
    avg_degree = 251927 / 3000       # 3000子图平均出度 ≈ 84

    # 稀疏 SNN: 每步 N*fr 个神经元放电, 每个触发 avg_degree 个突触「加法」
    syn_ops_per_step = N * fr * avg_degree
    total_ops_snn = syn_ops_per_step * T
    energy_snn = total_ops_snn * 0.9e-12   # J

    # 稠密 ANN: 每步 N×N 「乘加」(全连接)
    flops_per_step = 2 * N * N
    total_flops_ann = flops_per_step * T
    energy_ann = total_flops_ann * 3.7e-12  # J

    print(f"=== 能耗对比 (N={N}, T={T}, firing_rate={fr}) ===")
    print(f"稀疏 SNN: {total_ops_snn:.2e} 加法/样本 -> {energy_snn*1e9:.2f} nJ")
    print(f"稠密 ANN: {total_flops_ann:.2e} 乘加/样本 -> {energy_ann*1e9:.2f} nJ")
    print(f"能耗比 (ANN/SNN): {energy_ann/energy_snn:.1f}x")

    print(f"\n=== 敏感性: firing rate 对能耗比的影响 ===")
    for fr2 in [0.096, 0.05, 0.01, 0.005]:
        ops2 = N * fr2 * avg_degree * T
        e2 = ops2 * 0.9e-12
        print(f"  firing_rate={fr2:.3f}: 能耗 {e2*1e9:.2f} nJ, 比ANN省 {energy_ann/e2:.1f}x")

    print("\n注: 大脑真实 firing rate ~1%, 这是「数量级效率」的来源;")


if __name__ == '__main__':
    main()
