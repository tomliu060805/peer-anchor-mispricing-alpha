# -*- coding: utf-8 -*-
"""行业一致性检验的等稀疏度对照。

原工作的结构性证据是: 暴露相似图的"行业内/行业间平均边权比"为 5.30×(平静期),
远高于相关图的 2.42×, 并以此说明联合估计提取的局部结构具有经济含义。

问题在于对照不等价: **暴露图是稀疏的(L1 惩罚), 而相关矩阵是稠密的(所有元素非零)**。
图越稀疏, 留下的边越是最强的边, 而最强的边天然更可能落在同一行业内 ——
行业比会随稀疏度单调上升。稀疏图与稠密图直接比这个数, 比的可能是稀疏度而非结构。

本脚本在**每一个稀疏度**上做等边数对照: 把相关图按权重截断到与暴露图相同的边数,
再比两者的行业比。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(_HERE, '..'))
_sys.path.insert(0, _HERE)
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, importlib.util
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import mingle as MG

_spec = importlib.util.spec_from_file_location('rep', f'{_HERE}/10_mingle_replicate.py')
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)

DATES = ['2019-06-01', '2020-06-01', '2021-06-01', '2022-06-01',
         '2023-06-01', '2024-06-01', '2025-06-01']
BETA_MULT = [0.5, 2, 8, 20, 40, 80, 160]


def top_m(A, M):
    """只保留权重最大的 M 条边。"""
    N = len(A)
    iu = np.triu_indices(N, 1)
    v = A[iu]
    if M >= len(v):
        return A.copy()
    th = np.partition(v, -M)[-M]
    B = np.where(A >= th, A, 0.0)
    np.fill_diagonal(B, 0)
    return B


def main():
    d = rep._load()
    rb = rep.rebalance_days()
    ts = [int(rb[np.searchsorted(rb, np.searchsorted(d['dates'], x))]) for x in DATES]
    print(f'{len(ts)} 个截面 x {len(BETA_MULT)} 个稀疏度\n')
    print(f'{"beta×":>6s} {"平均度":>7s} {"边数":>7s} {"边密度":>7s} '
          f'{"暴露图行业比":>12s} {"相关图@等边数":>13s} {"暴露/相关":>9s}')
    rows = []
    for bm in BETA_MULT:
        acc = []
        for t in ts:
            uni = rep.universe_at(t)
            R = d['ret'][t - rep.TRAIN_D + 1:t + 1, uni].T.astype(np.float64)
            R = np.nan_to_num(R, nan=0.0)
            R = R - R.mean(1, keepdims=True)
            B0, _, _, _, _ = MG.fit_mingle(R, K=rep.K_FACTOR, halflife=rep.HALFLIFE,
                                           alpha=1.0, n_outer=40)
            sq = (B0 ** 2).sum(1)
            Zd = sq[:, None] + sq[None, :] - 2 * (B0 @ B0.T)
            np.maximum(Zd, 0, out=Zd)
            b0 = np.median(Zd[np.triu_indices(len(uni), 1)]) * 0.5
            B, F, W, Psi, info = MG.fit_mingle(R, K=rep.K_FACTOR, halflife=rep.HALFLIFE,
                                               alpha=1.0, beta=b0 * bm)
            Cr = np.abs(np.corrcoef(R))
            np.fill_diagonal(Cr, 0)
            iu = np.triu_indices(len(uni), 1)
            M = int((W[iu] > 1e-8).sum())
            if M < 200:
                continue
            acc.append((info['avg_degree'], M, info['edge_density'],
                        rep._industry_ratio(W, d['codes'][uni], d['dates'][t - 1]),
                        rep._industry_ratio(top_m(Cr, M), d['codes'][uni], d['dates'][t - 1])))
        if not acc:
            continue
        a = np.nanmean(np.array(acc, dtype=float), 0)
        rows.append({'beta_mult': bm, 'avg_degree': a[0], 'n_edge': a[1],
                     'density': a[2], 'ratio_exposure': a[3], 'ratio_corr_matched': a[4],
                     'rel': a[3] / a[4]})
        print(f'{bm:6.1f} {a[0]:7.2f} {a[1]:7.0f} {a[2]:7.3f} '
              f'{a[3]:12.2f} {a[4]:13.2f} {a[3]/a[4]:9.2f}')

    print('\n读法: 两条曲线都随稀疏度上升 —— 行业比本身就是稀疏度的函数。')
    print('      在可用于聚类的密度上(平均度 8-13), 暴露图一致低于相关图;')
    print('      只有在图接近断开(平均度<2)时才反超, 而那时的图无法支撑谱切分。')
    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump({'dates': DATES, 'rows': rows},
              open(f'{OUTPUT_DIR}/research/mingle_industry_sparsity.json', 'w'),
              ensure_ascii=False, indent=1, default=float)
    print('\n已写 output/research/mingle_industry_sparsity.json')


if __name__ == '__main__':
    main()
