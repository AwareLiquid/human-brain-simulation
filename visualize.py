"""visualize.py — SNN spike 活动可视化 (展示稀疏脉冲核心卖点)。

训练一个小 SNN → 记录推理一个样本时的逐层 spike 序列 → 生成自包含 HTML。
HTML 用 Canvas 画 spike raster + 实时动画, 直观展示「只有少数神经元在放电」。
"""
import torch
import numpy as np
from awareliquid import SparseSNN, longtail_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist():
    d = np.load('data/mnist.npz')
    Xtr = d['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
    ytr = d['y_train'].astype(np.int64)
    Xte = d['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
    yte = d['y_test'].astype(np.int64)
    return Xtr, ytr, Xte, yte


def build_and_train():
    import torch.nn.functional as F
    Xtr, ytr, _, _ = load_mnist()
    dims = [784, 300, 300]
    masks, signs = [], []
    for i in range(len(dims) - 1):
        m, s = longtail_mask(dims[i + 1], dims[i], 0.05)
        masks.append(m.to(device))
        signs.append(s.to(device))
    model = SparseSNN(dims, masks, signs, 10, T=15, inter_gain=5.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(Xtr[:10000], device=device)
    yt = torch.tensor(ytr[:10000], device=device)
    for ep in range(3):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            out, _ = model(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def record_spikes(model, x, T=15):
    """记录一个样本逐层的 spike 序列。返回 list[(T, N)] + 每层稀疏度。"""
    layers = []
    for layer in model.layers:
        W = (layer.w * layer.mask).detach()
        v = torch.zeros(1, layer.out_dim, device=device)
        spike_seq = []
        for t in range(T):
            I = x @ W.T
            v = v * layer.decay + I
            s = (v > layer.threshold).float()
            v = v * (1.0 - s)
            spike_seq.append(s.cpu().numpy()[0])
        sp = np.array(spike_seq)                     # (T, N)
        layers.append(sp)
        x = (sp.mean(axis=0, keepdims=True)) * model.inter_gain   # firing rate 传入下一层
        x = torch.tensor(x, device=device)
    return layers


def generate_html(layers, label, pred):
    T = layers[0].shape[0]
    # 每层取前 120 个神经元显示
    layers_disp = [L[:, :120] for L in layers]
    # 稀疏度
    sparsity = [float(1.0 - L.mean()) for L in layers]

    data_json = []
    for li, L in enumerate(layers_disp):
        # 转成 list of list (T x N)
        data_json.append({"name": f"Layer {li+1}", "spikes": L.tolist(),
                          "sparsity": sparsity[li]})

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>类脑稀疏脉冲 SNN — Spike 活动</title>
<style>
body{background:#0d1117;color:#c9d1d9;font-family:system-ui;margin:0;padding:20px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#8b949e;font-size:13px;margin-bottom:16px}
.layer{margin-bottom:28px}
.lname{font-size:14px;margin-bottom:6px}
canvas{background:#010409;border:1px solid #30363d;border-radius:6px}
.badge{display:inline-block;background:#1f6feb;color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:8px}
</style></head><body>
<h1>🧠 稀疏脉冲 SNN — 推理时的神经活动</h1>
<div class="sub">输入手写数字 <b>""" + str(label) + """</b> · 预测 <b>""" + str(pred) + """</b> · 白色点 = 神经元放电 · 注意只有极少数神经元在活跃（稀疏脉冲 → 省 2 个数量级能耗）</div>
<div id="root"></div>
<script>
const DATA = """ + str(data_json).replace("'", '"') + """;
const T = """ + str(T) + """;
const CELL = 4;
function drawLayer(layer, canvas){
  const ctx = canvas.getContext('2d');
  const N = layer.spikes[0].length;
  canvas.width = T * CELL;
  canvas.height = N * CELL;
  ctx.fillStyle = '#010409'; ctx.fillRect(0,0,canvas.width,canvas.height);
  for(let t=0;t<T;t++){
    for(let n=0;n<N;n++){
      if(layer.spikes[t][n] > 0){
        ctx.fillStyle = '#58a6ff';
        ctx.fillRect(t*CELL, n*CELL, CELL-1, CELL-1);
      }
    }
  }
}
const root = document.getElementById('root');
DATA.forEach(layer => {
  const div = document.createElement('div'); div.className='layer';
  const name = document.createElement('div'); name.className='lname';
  name.innerHTML = layer.name + ' <span class="badge">活跃神经元仅 ' + (100*(1-layer.sparsity)).toFixed(1) + '%</span>';
  const canvas = document.createElement('canvas');
  div.appendChild(name); div.appendChild(canvas); root.appendChild(div);
  drawLayer(layer, canvas);
});
</script></body></html>"""
    return html


def main():
    print("=== 训练小 SNN (2层300, 快速) ===")
    model = build_and_train()
    model.eval()

    Xtr, ytr, Xte, yte = load_mnist()
    sample = Xte[0]
    label = int(yte[0])

    with torch.no_grad():
        x = torch.tensor(sample.reshape(1, -1), device=device)
        layers = record_spikes(model, x)
        out, _ = model(torch.tensor(sample.reshape(1, -1), device=device))
        pred = int(out.argmax(1).item())

    html = generate_html(layers, label, pred)
    with open('visualize.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("生成 visualize.html")
    for i, L in enumerate(layers):
        print(f"  Layer {i+1}: 活跃神经元比例 = {L.mean()*100:.1f}%")


if __name__ == '__main__':
    main()
