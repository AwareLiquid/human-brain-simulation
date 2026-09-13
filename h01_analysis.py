"""h01_analysis.py — H01 人脑连接组统计规律 (对比果蝇 MaleCNS)。

验证「脑的普遍规律」(长尾度分布、E/I、稀疏度) 在人脑 H01 上是否也成立。
"""
import numpy as np

m = np.load('data/h01/metadata.npz', allow_pickle=True)
positions = m['positions']
is_exc = m['is_excitatory']
is_inh = m['is_inhibitory']
layers = m['layers']
cell_types = m['cell_types']

e = np.load('data/h01/edges.npz', allow_pickle=True)
edges = e['edges']          # (116611, 3) [pre, post, type]

n = len(positions)
n_edges = len(edges)
density = n_edges / (n * n)

print("=== H01 人脑连接组 (1mm³ 颞叶皮层) ===")
print(f"神经元: {n}")
print(f"连接边: {n_edges}")
print(f"密度: {density:.6f}")
print(f"兴奋: {is_exc.sum()} ({is_exc.mean():.1%})")
print(f"抑制: {is_inh.sum()} ({is_inh.mean():.1%})")

# 度分布
pre, post = edges[:, 0], edges[:, 1]
out_deg = np.bincount(pre, minlength=n)
in_deg = np.bincount(post, minlength=n)
for name, d in [('out', out_deg), ('in', in_deg)]:
    dz = d[d > 0]
    print(f"{name}度: mean={d.mean():.2f} median={np.median(dz):.1f} "
          f"max={d.max()} p90={np.percentile(dz, 90):.0f} p99={np.percentile(dz, 99):.0f}")

# 皮层层次分布
print("\n皮层层次分布:")
for l in np.unique(layers):
    print(f"  {l}: {(layers == l).sum()}")

# 细胞类型分布
print("\n细胞类型分布:")
for ct in np.unique(cell_types):
    print(f"  {ct}: {(cell_types == ct).sum()}")

# 3D 范围 (用于可视化)
print(f"\n3D 位置范围 (nm):")
for i, ax in enumerate(['x', 'y', 'z']):
    print(f"  {ax}: [{positions[:, i].min():.0f}, {positions[:, i].max():.0f}]")

# 对比果蝇 MaleCNS
print("\n=== 对比: 果蝇 MaleCNS vs 人脑 H01 ===")
print(f"{'':20s} {'MaleCNS(果蝇)':>16s} {'H01(人脑)':>16s}")
print(f"{'神经元':20s} {'165,122':>16s} {n:>16,}")
print(f"{'密度':20s} {'0.090%':>16s} {density:>16.4%}")
print(f"{'E/I 比例':20s} {'60/40':>16s} {f'{is_exc.mean():.0%}/{is_inh.mean():.0%}':>16s}")
print(f"{'平均度':20s} {'148.6':>16s} {out_deg.mean():>16.2f}")
print(f"{'最大度':20s} {'11203':>16s} {out_deg.max():>16}")
