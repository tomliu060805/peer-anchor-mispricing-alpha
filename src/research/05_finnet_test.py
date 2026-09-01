# -*- coding: utf-8 -*-
"""财务结构网络的两个决定性检验。

前置结论(03 号脚本): 财务结构聚类在留一维度与跨期两项非循环判据下都成立,
且与行业近乎正交。但"财务结构相近"与"价格一起动"是两个对象——
能不能进五锚, 取决于下面两条, 与前面的复现结论无关:

  1. 正交性   与价格/价量/转移熵三张网络的邻居重叠。重叠高 = 只是换个方式
              找到同一批邻居, 没有新信息。
  2. 预测力   由该网络算出的动量 gap 的周频 RankIC。以及与现有五锚合成后
              是否还有增量(对现有 base_score 做正交化后的残差 IC)。

分段: dev/val 分开报告, test 不碰。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np

W, K, MOM, WLIM = 120, 5, 20, 250
SEG = {'dev': ('2016-01-01', '2022-12-31'), 'val': ('2023-01-01', '2024-08-16')}


def rank_xs(x):
    m = ~np.isnan(x)
    r = np.full_like(x, np.nan, dtype=np.float32)
    if m.sum() < 50:
        return r
    rr = np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m] = (rr - rr.mean()) / (rr.std() + 1e-12)
    return r


def gap(z, t, mom):
    RB, NB, WV = z['t'], z['n'], z['w']
    b = np.searchsorted(RB, t, side='right') - 1
    N = NB.shape[1]
    g = np.full(N, np.nan, np.float32)
    if b < 0:
        return g
    nbr, wgt = NB[b], np.maximum(WV[b], 0)
    idx = np.where((nbr[:, 0] >= 0) & (wgt.sum(1) > 1e-12))[0]
    if len(idx) < 100:
        return g
    nb, w0 = nbr[idx], wgt[idx]
    pm = mom[np.where(nb >= 0, nb, 0)]
    msk = (nb >= 0) & ~np.isnan(pm)
    w = w0 * msk
    ok = w.sum(1) > 1e-12
    mu = (np.nan_to_num(pm) * w).sum(1) / np.maximum(w.sum(1), 1e-12)
    g[idx[ok]] = mu[ok] - mom[idx[ok]]
    return g


def overlap(a, b):
    """两张网络逐股邻居集合的平均重叠率(按重建点对齐)。"""
    Ra, Rb = a['t'], b['t']
    tot = hit = 0
    for i, t in enumerate(Ra):
        j = int(np.searchsorted(Rb, t))
        if j >= len(Rb) or Rb[j] != t:
            continue
        na, nb = a['n'][i], b['n'][j]
        m = (na[:, 0] >= 0) & (nb[:, 0] >= 0)
        x, y = na[m], nb[m]
        if len(x) == 0:
            continue
        # 向量化: 对每只股票, 计数 x 的邻居中有多少出现在 y 的邻居里
        eq = (x[:, :, None] == y[:, None, :]) & (x[:, :, None] >= 0)
        hit += int(eq.any(2).sum())
        tot += int((x >= 0).sum())
    return hit / max(tot, 1)


def main():
    g = np.load(f'{CACHE}/daily_grid.npz')
    ret, dates, close = g['ret'], g['dates'], g['close']
    paused, at_hl, at_ll = g['paused'], g['at_hlimit'], g['at_llimit']
    T, N = ret.shape
    st = np.load(f'{CACHE}/st_grid.npz')['is_st']
    r0 = np.nan_to_num(ret, nan=0.0)
    logc = np.cumsum(np.log1p(r0), 0)
    mom20 = np.full((T, N), np.nan, np.float32)
    mom20[MOM:] = (logc[MOM:] - logc[:-MOM]).astype(np.float32)
    c2 = np.cumsum(r0 ** 2, 0)
    vol20 = np.full((T, N), np.nan, np.float32)
    vol20[MOM:] = np.sqrt((c2[MOM:] - c2[:-MOM]) / MOM).astype(np.float32)
    tradable = (paused < 0.5) & (at_hl < 0.5) & (at_ll < 0.5) & (st < 0.5) & ~np.isnan(ret)
    chl = np.cumsum(np.nan_to_num(at_hl, nan=0.0), 0)
    lim250 = np.zeros((T, N), np.float32)
    lim250[WLIM:] = chl[WLIM:] - chl[:-WLIM]

    nets = {k: np.load(f'{CACHE}/nets_{k}.npz') for k in ('price', 'dual', 'te', 'fin')}
    FI = np.load(f'{CACHE}/fi_grid.npz')['fi']
    ROE = FI[:, 0]

    print('=== 1. 邻居重叠率 (财务网络 vs 其余三张) ===')
    ov = {}
    for k in ('price', 'dual', 'te'):
        ov[k] = overlap(nets['fin'], nets[k])
        print(f'  fin vs {k:6s} {ov[k]:.1%}')
    print(f'  (参考: 项目已知 te vs price 重叠 {overlap(nets["te"], nets["price"]):.1%})')

    print('\n=== 2. 周频 RankIC (下一周收益) ===')
    sig = [int(x) for x in np.arange(WLIM + 1, T - 6, 5)]
    rows = {k: {s: [] for s in SEG} for k in ('price', 'dual', 'te', 'fin', 'roe')}
    resid_ic = {s: [] for s in SEG}
    for si, t in enumerate(sig[:-1]):
        ds = np.datetime64(str(dates[t]))
        seg = next((s for s, (a, b) in SEG.items()
                    if np.datetime64(a) <= ds <= np.datetime64(b)), None)
        if seg is None:
            continue
        t1 = sig[si + 1]
        fwd = np.exp(logc[t1] - logc[t]) - 1
        dom = (lim250[t] >= 2) & tradable[t]
        v = np.maximum(vol20[t] * np.sqrt(MOM), 1e-4)
        gs = {k: rank_xs(gap(nets[k], t, mom20[t]) / v) for k in ('price', 'dual', 'te', 'fin')}
        gs['roe'] = rank_xs(-gap(nets['price'], t, ROE[t]))       # 自身-邻居
        m0 = dom & np.isfinite(fwd)
        for k, x in gs.items():
            m = m0 & np.isfinite(x)
            if m.sum() > 200:
                rows[k][seg].append(np.corrcoef(
                    np.argsort(np.argsort(x[m])), np.argsort(np.argsort(fwd[m])))[0, 1])
        # 财务锚对现有五锚合成分的正交化残差 IC
        base = np.nanmean(np.stack([gs['price'], gs['dual'], gs['te'], gs['roe']]), 0)
        m = m0 & np.isfinite(gs['fin']) & np.isfinite(base)
        if m.sum() > 200:
            b, a_ = np.polyfit(base[m], gs['fin'][m], 1)
            r = gs['fin'][m] - (b * base[m] + a_)
            resid_ic[seg].append(np.corrcoef(
                np.argsort(np.argsort(r)), np.argsort(np.argsort(fwd[m])))[0, 1])

    def rep(v):
        v = np.array(v)
        return f'{v.mean():+.4f} (ICIR {v.mean()/(v.std()+1e-12)*np.sqrt(len(v)):5.2f}, n={len(v)})'

    print(f'{"锚":8s} {"dev":>34s} {"val":>34s}')
    for k in ('price', 'dual', 'te', 'roe', 'fin'):
        print(f'{k:8s} {rep(rows[k]["dev"]):>34s} {rep(rows[k]["val"]):>34s}')
    print('\n=== 3. 财务锚对现有四锚正交化后的残差 IC ===')
    for s in SEG:
        print(f'  {s:4s} {rep(resid_ic[s])}')

    out = {'overlap': ov,
           'ic': {k: {s: float(np.mean(rows[k][s])) for s in SEG} for k in rows},
           'icir': {k: {s: float(np.mean(rows[k][s]) / (np.std(rows[k][s]) + 1e-12)
                                 * np.sqrt(len(rows[k][s]))) for s in SEG} for k in rows},
           'resid_ic': {s: float(np.mean(resid_ic[s])) for s in SEG},
           'resid_icir': {s: float(np.mean(resid_ic[s]) / (np.std(resid_ic[s]) + 1e-12)
                                   * np.sqrt(len(resid_ic[s]))) for s in SEG}}
    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump(out, open(f'{OUTPUT_DIR}/research/finnet_test.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\n已写 output/research/finnet_test.json')


if __name__ == '__main__':
    main()
