"""探查 H01 人脑连接组数据 (NathanRoll/h01-cortex-snn)。"""
import numpy as np
import json

print("=== metadata.npz ===")
m = np.load('data/h01/metadata.npz', allow_pickle=True)
for k in m.files:
    v = m[k]
    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    if v.ndim == 1 and v.shape[0] < 20:
        print(f"    values={v[:5]}")

print("\n=== config.json ===")
with open('data/h01/config.json') as f:
    cfg = json.load(f)
print(json.dumps(cfg, indent=2)[:2000])

print("\n=== somas_filtered.csv (前3行 + 列) ===")
with open('data/h01/somas_filtered.csv') as f:
    lines = f.readlines()
print("列:", lines[0].strip())
for l in lines[1:4]:
    print(l.strip()[:200])

print("\n=== edges.npz ===")
e = np.load('data/h01/edges.npz', allow_pickle=True)
for k in e.files:
    v = e[k]
    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    print(f"    前5: {v[:5]}")
