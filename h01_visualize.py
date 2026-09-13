"""h01_visualize.py — 人脑神经元 3D 可视化 (16k 神经元, E/I 着色, 可旋转)。

生成自包含 HTML, 用 Canvas 2D 做 3D 旋转投影, 展示真实人脑皮层神经元的空间分布。
兴奋神经元=红, 抑制神经元=蓝, 鼠标拖拽旋转。
"""
import numpy as np
import json

m = np.load('data/h01/metadata.npz', allow_pickle=True)
pos = m['positions'].astype(np.float64)
is_exc = m['is_excitatory']
is_inh = m['is_inhibitory']

# 归一化到 [-1, 1] (各轴独立, 展示皮层层次结构)
for i in range(3):
    mn, mx = pos[:, i].min(), pos[:, i].max()
    if mx > mn:
        pos[:, i] = (pos[:, i] - mn) / (mx - mn) * 2 - 1

# 颜色: 兴奋=红, 抑制=蓝, 其他=灰
colors = []
for e, inh in zip(is_exc, is_inh):
    if e:
        colors.append([1.0, 0.35, 0.3])
    elif inh:
        colors.append([0.3, 0.55, 1.0])
    else:
        colors.append([0.75, 0.75, 0.75])

data = {"points": pos.tolist(), "colors": colors}

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>人脑皮层神经元 3D — H01</title>
<style>
body{background:#0d1117;color:#c9d1d9;font-family:system-ui;margin:0;padding:16px;overflow:hidden}
h1{font-size:18px;margin:0 0 4px}
.sub{color:#8b949e;font-size:13px;margin-bottom:10px}
canvas{display:block;cursor:grab}
.legend{position:fixed;bottom:16px;left:16px;font-size:13px;background:#161b22;padding:8px 12px;border-radius:6px;border:1px solid #30363d}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
</style></head><body>
<h1>🧠 真实人脑皮层神经元 — 3D 分布 (H01, 1mm³ 颞叶皮层)</h1>
<div class="sub">16,087 个神经元 · 拖拽旋转 · 滚轮缩放 · <span style="color:#ff6b6b">兴奋神经元</span> / <span style="color:#6bb5ff">抑制神经元</span></div>
<canvas id="c"></canvas>
<div class="legend">
  <span class="dot" style="background:#ff6b6b"></span>兴奋 (10,531) &nbsp;
  <span class="dot" style="background:#6bb5ff"></span>抑制 (4,688) &nbsp;
  <span class="dot" style="background:#bfbfbf"></span>未分类
</div>
<script>
const DATA = """ + json.dumps(data) + """;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let rx = 0.5, ry = 0.0, scale = 260, cx = canvas.width/2, cy = canvas.height/2;
let dragging = false, lx = 0, ly = 0;

function project(p){
  const cosY = Math.cos(ry), sinY = Math.sin(ry);
  const cosX = Math.cos(rx), sinX = Math.sin(rx);
  let x = p[0]*cosY + p[2]*sinY;
  let z = -p[0]*sinY + p[2]*cosY;
  let y = p[1]*cosX - z*sinX;
  z = p[1]*sinX + z*cosX;
  return [x*scale + cx, y*scale + cy, z];
}

function draw(){
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,canvas.width,canvas.height);
  // 深度排序 (远先画)
  const pts = DATA.points.map((p,i)=>({p, z:project(p)[2], i})).sort((a,b)=>a.z-b.z);
  for(const item of pts){
    const [x,y] = project(item.p);
    const c = DATA.colors[item.i];
    ctx.fillStyle = `rgb(${(c[0]*255)|0},${(c[1]*255)|0},${(c[2]*255)|0})`;
    ctx.fillRect(x, y, 2, 2);
  }
}

canvas.addEventListener('mousedown', e=>{dragging=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mousemove', e=>{
  if(!dragging) return;
  ry += (e.clientX-lx)*0.008;
  rx += (e.clientY-ly)*0.008;
  lx=e.clientX;ly=e.clientY;
  draw();
});
window.addEventListener('mouseup', ()=>{dragging=false;});
canvas.addEventListener('wheel', e=>{e.preventDefault(); scale *= e.deltaY>0?0.9:1.1; draw();});

draw();
</script></body></html>"""

with open('h01_visualize.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("生成 h01_visualize.html")
print(f"神经元: {len(pos)}, 兴奋: {is_exc.sum()}, 抑制: {is_inh.sum()}")
print(f"3D 范围 (归一化后): x[{pos[:,0].min():.2f},{pos[:,0].max():.2f}] "
      f"y[{pos[:,1].min():.2f},{pos[:,1].max():.2f}] z[{pos[:,2].min():.2f},{pos[:,2].max():.2f}]")
