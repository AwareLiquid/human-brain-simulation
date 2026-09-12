"""pipeline.py — MaleCNS feather → 带递质符号的稀疏邻接矩阵。

复用 philshiu (Nature 2024) 的做法:
  权重 = 突触数 × 递质符号 (兴奋 +1 / 抑制 -1)
输出: scipy.sparse.csr_matrix + body_id 列表, 存 npz/npy。
"""
import numpy as np
import scipy.sparse as sp
import pyarrow.feather as fe

# 果蝇中枢神经系统递质符号惯例
# 兴奋: acetylcholine
# 抑制: gaba, glutamate(中枢), histamine
# 调制: dopamine, serotonin, octopamine -> 0 (不作为固定权重, 后续当外部调制信号)
NT_SIGN = {
    'acetylcholine': +1.0,
    'gaba': -1.0,
    'glutamate': -1.0,
    'histamine': -1.0,
    'dopamine': 0.0,
    'serotonin': 0.0,
    'octopamine': 0.0,
}


def load():
    cw = fe.read_table("data/connectome-weights-male-cns-v1.0-minconf-0.5.feather").to_pandas()
    nt = fe.read_table("data/body-neurotransmitters-male-cns-v1.0.feather").to_pandas()
    ann = fe.read_table("data/body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
    return cw, nt, ann


def main():
    cw, nt, ann = load()

    print("=== status 分布 ===")
    print(ann['status'].value_counts(dropna=False).head(20))

    print("\n=== consensus_nt 分布 ===")
    print(nt['consensus_nt'].value_counts(dropna=False).head(20))

    # 有效神经元: Traced (proofread 完成)
    valid = ann[ann['status'] == 'Traced']['bodyId'].values
    valid_set = set(valid.tolist())
    print(f"\nTraced 神经元数: {len(valid)}")

    # body -> 递质符号 (vectorized)
    nt2 = nt[nt['body'].isin(valid_set)][['body', 'consensus_nt']].drop_duplicates('body')
    nt_map = nt2.set_index('body')['consensus_nt'].to_dict()

    # 过滤边: pre/post 都在有效神经元集合
    e = cw[cw['body_pre'].isin(valid_set) & cw['body_post'].isin(valid_set)]
    print(f"有效边数(两端都 Traced): {len(e)}")

    # body -> index 映射
    body_ids = valid
    idx = {b: i for i, b in enumerate(body_ids)}

    # 符号化权重 (vectorized, 避免 Python 循环)
    e = e.copy()
    e['sign'] = e['body_pre'].map(nt_map).map(NT_SIGN).fillna(0.0).astype(np.float32)
    e['w_signed'] = e['weight'].astype(np.float32) * e['sign']
    e = e[e['sign'] != 0.0]

    rows = e['body_pre'].map(idx).values
    cols = e['body_post'].map(idx).values
    vals = e['w_signed'].values

    n = len(body_ids)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.sum_duplicates()

    print(f"\n邻接矩阵: {A.shape}, nnz={A.nnz}, 密度={A.nnz/(A.shape[0]*A.shape[1]):.6f}")

    sp.save_npz("data/malecns_signed_adj.npz", A)
    np.save("data/malecns_body_ids.npy", body_ids)
    print("saved: data/malecns_signed_adj.npz + data/malecns_body_ids.npy")

    deg = np.asarray(A.getnnz(axis=1)).ravel()
    print(f"\n度数: mean={deg.mean():.1f}, median={np.median(deg):.1f}, max={deg.max()}, 有连接的神经元比例={np.mean(deg > 0):.3f}")


if __name__ == '__main__':
    main()
