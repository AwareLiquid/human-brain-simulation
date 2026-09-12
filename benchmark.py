"""benchmark.py — 多层稀疏 SNN: 深度收益 + 真稀疏算力实测。"""
import torch
import torch.nn.functional as F
import time
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


def build_model(dims, density, num_classes=10, T=15, inter_gain=1.0):
    masks, signs = [], []
    for i in range(len(dims) - 1):
        m, s = longtail_mask(dims[i + 1], dims[i], density)
        masks.append(m.to(device))
        signs.append(s.to(device))
    return SparseSNN(dims, masks, signs, num_classes, T=T, inter_gain=inter_gain).to(device)


def train(model, Xtr, ytr, epochs, batch=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out, _ = model(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {ep+1}: loss={total/n:.4f}")


def evaluate(model, Xte, yte):
    model.eval()
    Xt = torch.tensor(Xte, device=device)
    yt = torch.tensor(yte, device=device)
    with torch.no_grad():
        out, _ = model(Xt)
        return float((out.argmax(1) == yt).float().mean().item())


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    n_train, epochs = 20000, 8
    Xb = torch.tensor(Xte[:2000], device=device)

    print("=== 单层 SNN (784→800→10) ===")
    m1 = build_model([784, 800], 0.05)
    t0 = time.time()
    train(m1, Xtr[:n_train], ytr[:n_train], epochs)
    a1 = evaluate(m1, Xte[:2000], yte[:2000])
    print(f"  准确率={a1:.4f}  总稀疏度={m1.total_sparsity:.3f}  用时={time.time()-t0:.0f}s")

    print("\n=== 两层 SNN (784→800→800→10) ===")
    m2 = build_model([784, 800, 800], 0.05, inter_gain=5.0)
    t0 = time.time()
    train(m2, Xtr[:n_train], ytr[:n_train], epochs)
    a2 = evaluate(m2, Xte[:2000], yte[:2000])
    print(f"  准确率={a2:.4f}  总稀疏度={m2.total_sparsity:.3f}  用时={time.time()-t0:.0f}s")

    # 实测 sparse vs dense 推理算力
    print("\n=== 实测: sparse vs dense 推理算力 (单层, 2000 样本) ===")
    m1.eval()
    with torch.no_grad():
        # warmup
        _ = m1(Xb)
        _ = m1.forward_sparse(Xb)
        t0 = time.time()
        for _ in range(3):
            _ = m1(Xb)
        t_dense = (time.time() - t0) / 3
        t0 = time.time()
        for _ in range(3):
            _ = m1.forward_sparse(Xb)
        t_sparse = (time.time() - t0) / 3
    print(f"  dense 推理: {t_dense*1000:.1f} ms")
    print(f"  sparse 推理: {t_sparse*1000:.1f} ms")
    print(f"  加速比: {t_dense/t_sparse:.2f}x (sparse 更快则 >1)")


if __name__ == '__main__':
    main()
