"""engine.py — LIF 储备池引擎 + 三种拓扑生成器。

复用 philshiu (Nature 2024) 的 LIF 方程与参数思路, 但用 PyTorch 实现,
以便对拍实验灵活换连接结构。

对拍核心: 三种拓扑「同密度 / 同 E-I 比例 / 同谱半径」, 只隔离「连接结构」这一个变量。
储备池权重固定, 只训练线性读出层。
"""
import numpy as np
import torch
import scipy.sparse as sp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_torch_csr(A):
    A = A.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.copy()).long(),
        torch.from_numpy(A.indices.copy()).long(),
        torch.from_numpy(A.data.copy()).float(),
        size=A.shape).to(device)


def spectral_radius(A):
    """power iteration 估计谱半径。"""
    v = np.random.default_rng(0).standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(200):
        w = A.dot(v)
        n = np.linalg.norm(w)
        if n < 1e-12:
            return 0.0
        v = w / n
    return float(np.abs(np.dot(v, A.dot(v))))


def extract_subgraph(A, k):
    """度数 top-k 神经元的诱导子图 (A 为 csr)。"""
    deg = np.asarray(A.getnnz(axis=1)).ravel()
    top = np.argsort(-deg)[:k]
    return A[top][:, top].tocsr()


def _block_sparse(n, n_blocks, density, rng, e_ratio, inter_frac=0.1, seed=0):
    """MoE 式结构化稀疏: 分块对角(组内稠密) + 跨块稀疏路由。"""
    block = n // n_blocks
    intra = min(0.9, density * n_blocks * (1 - inter_frac))
    blks = []
    for b in range(n_blocks):
        bm = sp.random(block, block, density=intra, format='csr',
                       random_state=seed + b)
        blks.append(bm)
    A = sp.block_diag(blks).tocsr()
    if A.shape[0] < n:
        A = sp.vstack([A, sp.csr_matrix((n - A.shape[0], A.shape[1]))])
        A = sp.hstack([A, sp.csr_matrix((A.shape[0], n - A.shape[1]))]).tocsr()
    inter_nnz = int(n * n * density * inter_frac)
    if inter_nnz > 0:
        r = rng.integers(0, n, inter_nnz)
        c = rng.integers(0, n, inter_nnz)
        A = (A + sp.csr_matrix((np.ones(inter_nnz), (r, c)), shape=(n, n))).tocsr()
    A.data = np.where(rng.random(A.nnz) < e_ratio, 1.0, -1.0).astype(np.float32)
    return A


def make_topologies(A_real, seed=42):
    """返回 dict: {'real','random','block'} -> 带符号 csr。
    三组同密度 / 同 E-I 比例 / 同谱半径(0.9), 只隔离连接结构。
    """
    rng = np.random.default_rng(seed)
    n = A_real.shape[0]
    nnz = A_real.nnz
    density = nnz / (n * n)
    e_ratio = float(np.mean(A_real.data > 0)) if nnz else 0.8

    A_real = A_real.copy()

    # random: ER 同密度
    A_rand = sp.random(n, n, density=density, format='csr', random_state=seed)
    A_rand.data = np.where(rng.random(A_rand.nnz) < e_ratio, 1.0, -1.0).astype(np.float32)

    # block: MoE 式结构化稀疏
    A_block = _block_sparse(n, 16, density, rng, e_ratio, seed=seed)

    out = {}
    for name, A in [('real', A_real), ('random', A_rand), ('block', A_block)]:
        r = spectral_radius(A)
        if r > 0:
            A = A * (0.9 / r)
        out[name] = A.astype(np.float32)
    return out


class LIFReservoir:
    """固定权重 LIF 储备池。输出 T 步内的平均 firing rate。"""

    def __init__(self, W, W_in, tau=20.0, threshold=1.0):
        self.W = W                      # torch sparse csr (n, n)
        self.W_in = W_in                # torch tensor (n, d_in)
        self.n = W.shape[0]
        self.decay = float(np.exp(-1.0 / tau))
        self.threshold = threshold

    def run(self, x, T):
        """x: (B, d_in) 输入电流 -> firing rate (B, n)。"""
        B = x.shape[0]
        I_in = x @ self.W_in.T          # (B, n)
        v = torch.zeros(B, self.n, device=device)
        rate = torch.zeros(B, self.n, device=device)
        s = torch.zeros(B, self.n, device=device)
        for _ in range(T):
            I_rec = torch.sparse.mm(self.W, s.T).T   # (B, n)
            v = v * self.decay + I_in + I_rec
            s = (v > self.threshold).float()
            v = v * (1.0 - s)           # hard reset
            rate = rate + s
        return rate / T

    def run_spikes(self, S, T, gain=1.0):
        """S: (B, d_in, T) 输入脉冲序列 -> firing rate (B, n)。gain 缩放输入电流。"""
        B = S.shape[0]
        v = torch.zeros(B, self.n, device=device)
        rate = torch.zeros(B, self.n, device=device)
        s = torch.zeros(B, self.n, device=device)
        for t in range(T):
            I_in = (S[:, :, t] @ self.W_in.T) * gain   # (B, n)
            I_rec = torch.sparse.mm(self.W, s.T).T     # (B, n)
            v = v * self.decay + I_in + I_rec
            s = (v > self.threshold).float()
            v = v * (1.0 - s)
            rate = rate + s
        return rate / T
