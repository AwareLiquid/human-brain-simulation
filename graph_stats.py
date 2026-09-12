"""graph_stats.py — 实验D: 提取 MaleCNS 拓扑的统计规律。

产出「设计新 SNN 架构的规律依据」: 稀疏度、度分布形状(是否幂律)、
E/I 比例、权重长尾、小世界性(聚类系数 vs 随机图)。
"""
import numpy as np
import scipy.sparse as sp


def degree_stats(A):
    out_deg = np.asarray(A.getnnz(axis=1)).ravel()
    in_deg = np.asarray(A.getnnz(axis=0)).ravel()
    for name, d in [('out', out_deg), ('in', in_deg)]:
        print(f"  {name}度: mean={d.mean():.1f} median={np.median(d):.1f} "
              f"max={d.max()} std={d.std():.1f} p90={np.percentile(d, 90):.0f} "
              f"p99={np.percentile(d, 99):.0f}")


def main():
    A = sp.load_npz('data/malecns_signed_adj.npz')
    n, nnz = A.shape[0], A.nnz
    print(f"=== MaleCNS 全脑 ===")
    print(f"神经元={n}, 连接边={nnz}, 密度={nnz/(n*n):.6f}")

    exc = int((A.data > 0).sum())
    inh = int((A.data < 0).sum())
    print(f"E/I 比例: 兴奋={exc/nnz:.3f} 抑制={inh/nnz:.3f}")

    print("度数分布:")
    degree_stats(A)

    w = np.abs(A.data)
    print(f"权重(突触数): mean={w.mean():.2f} median={np.median(w):.0f} "
          f"max={w.max():.0f} p90={np.percentile(w, 90):.0f} p99={np.percentile(w, 99):.0f}")

    # 3000 子图的小世界性
    from engine import extract_subgraph
    A_sub = extract_subgraph(A, 3000)
    k = 3000
    print(f"\n=== 3000 hub 诱导子图 ===")
    print(f"密度={A_sub.nnz/(k*k):.5f}, 平均度={A_sub.nnz/k:.1f}")

    import networkx as nx
    G = nx.from_scipy_sparse_array(A_sub, create_using=nx.DiGraph)
    Gu = G.to_undirected()
    cc = nx.average_clustering(Gu)
    print(f"真实拓扑 平均聚类系数: {cc:.4f}")

    # ER 随机图对照 (同密度)
    A_er = sp.random(k, k, density=A_sub.nnz/(k*k), format='csr', random_state=0)
    Ge = nx.from_scipy_sparse_array(A_er, create_using=nx.DiGraph).to_undirected()
    cc_er = nx.average_clustering(Ge)
    print(f"ER随机图(同密度) 聚类系数: {cc_er:.4f}")
    print(f"小世界比值 (真实聚类/随机聚类): {cc/max(cc_er, 1e-9):.2f}x  (>>1 表示小世界)")

    # 平均路径长度 (最大弱连通分量)
    largest = max(nx.weakly_connected_components(G), key=len)
    Gl = G.subgraph(largest)
    if len(Gl) > 2:
        apl = nx.average_shortest_path_length(Gl.to_undirected())
        print(f"最大弱连通分量={len(Gl)}/{k}, 平均路径长度={apl:.3f}")


if __name__ == '__main__':
    main()
