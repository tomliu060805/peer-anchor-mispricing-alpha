# -*- coding: utf-8 -*-
"""决定性检验: 财务网络的 IC 是不是自身反转的马甲。

问题
----
gap = 邻居动量加权均值 − 自身动量。若邻居对价格而言近乎随机, 邻居均值就约等于
全域均值(一个几乎不随股票变化的常数), 于是 gap ≈ 常数 − 自身动量, 排序等价于
**纯 20 日反转**。A股反转本就强, 05 号脚本里 fin 的 IC 0.064 可能全部由此而来,
与"财务结构"无关。

三条判据
--------
  1. 反转对照      纯反转因子 rank(-mom20/v) 自己的 IC 是多少
  2. 随机网络零基准 K=5 邻居从同一批可用股票里随机抽, 其余完全相同。
                   若随机网络给出同样的 IC, 那么信息全在自身动量项, 连边无贡献。
  3. 反转正交化    各锚对纯反转因子回归后, 残差还剩多少 IC

零基准跑 20 次取分布, 报告 fin 超出随机网络多少个标准差。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np

W, K, MOM, WLIM = 120, 5, 20, 250
SEG = {'dev': ('2016-01-01', '2022-12-31'), 'val': ('2023-01-01', '2024-08-16')}
N_RAND = 20
rng = np.random.default_rng(20260902)


def rank_xs(x):
    m = ~np.isnan(x)
    r = np.full_like(x, np.nan, dtype=np.float32)
    if m.sum() < 50:
        return r
    rr = np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m] = (rr - rr.mean()) / (rr.std() + 1e-12)
    return r


def gap_from(nbr, wgt, mom, N):
    g = np.full(N, np.nan, np.float32)
    idx = np.where((nbr[:, 0] >= 0) & (wgt.sum(1) > 1e-12))[0]
    if len(idx) < 100:
        return g
    nb, w0 = nbr[idx], np.maximum(wgt[idx], 0)
    pm = mom[np.where(nb >= 0, nb, 0)]
    msk = (nb >= 0) & ~np.isnan(pm)
    w = w0 * msk
    ok = w.sum(1) > 1e-12
    mu = (np.nan_to_num(pm) * w).sum(1) / np.maximum(w.sum(1), 1e-12)
    g[idx[ok]] = mu[ok] - mom[idx[ok]]
    return g


def ric(x, fwd, m):
    m = m & np.isfinite(x)
    if m.sum() < 200:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(x[m])),
                             np.argsort(np.argsort(fwd[m])))[0, 1])


def orth(x, base, m):
    """x 对 base 线性正交化后的残差(仅在 m 上)。"""
    mm = m & np.isfinite(x) & np.isfinite(base)
    if mm.sum() < 200:
        return None, None
    b, a = np.polyfit(base[mm], x[mm], 1)
    r = np.full_like(x, np.nan)
    r[mm] = x[mm] - (b * base[mm] + a)
    return r, mm


def main():
    g = np.load(f'{CACHE}/daily_grid.npz')
    ret, dates = g['ret'], g['dates']
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
    sig = [int(x) for x in np.arange(WLIM + 1, T - 6, 5)]

    acc = {k: {s: [] for s in SEG} for k in
           ('rev', 'price', 'dual', 'te', 'fin', 'rand')}
    accr = {k: {s: [] for s in SEG} for k in ('price', 'dual', 'te', 'fin', 'rand')}
    rand_runs = {s: [[] for _ in range(N_RAND)] for s in SEG}

    for si, t in enumerate(sig[:-1]):
        ds = np.datetime64(str(dates[t]))
        seg = next((s for s, (a, b) in SEG.items()
                    if np.datetime64(a) <= ds <= np.datetime64(b)), None)
        if seg is None:
            continue
        t1 = sig[si + 1]
        fwd = np.exp(logc[t1] - logc[t]) - 1
        dom = (lim250[t] >= 2) & tradable[t]
        m0 = dom & np.isfinite(fwd)
        v = np.maximum(vol20[t] * np.sqrt(MOM), 1e-4)
        rev = rank_xs(-mom20[t] / v)
        r_ = ric(rev, fwd, m0)
        if r_ is not None:
            acc['rev'][seg].append(r_)

        for k in ('price', 'dual', 'te', 'fin'):
            z = nets[k]
            b = np.searchsorted(z['t'], t, side='right') - 1
            if b < 0:
                continue
            x = rank_xs(gap_from(z['n'][b], z['w'][b], mom20[t], N) / v)
            r_ = ric(x, fwd, m0)
            if r_ is not None:
                acc[k][seg].append(r_)
            xr, mm = orth(x, rev, m0)
            if xr is not None:
                r2 = ric(xr, fwd, mm)
                if r2 is not None:
                    accr[k][seg].append(r2)

        # 随机网络: 邻居从 fin 网络当期有效的股票池里随机抽, 权重同为 1
        zf = nets['fin']
        b = np.searchsorted(zf['t'], t, side='right') - 1
        if b < 0:
            continue
        pool = np.where(zf['n'][b][:, 0] >= 0)[0]
        if len(pool) < 300:
            continue
        for ri in range(N_RAND):
            nb = rng.choice(pool, size=(N, K))
            nbr = np.full((N, K), -1, np.int32)
            nbr[pool] = nb[pool]
            wgt = (nbr >= 0).astype(np.float32)
            x = rank_xs(gap_from(nbr, wgt, mom20[t], N) / v)
            r_ = ric(x, fwd, m0)
            if r_ is not None:
                rand_runs[seg][ri].append(r_)
            if ri == 0:
                acc['rand'][seg].append(r_ if r_ is not None else np.nan)
                xr, mm = orth(x, rev, m0)
                if xr is not None:
                    r2 = ric(xr, fwd, mm)
                    if r2 is not None:
                        accr['rand'][seg].append(r2)

    def f(v):
        v = np.array([x for x in v if x is not None and np.isfinite(x)])
        if not len(v):
            return '  --', 0.0
        return f'{v.mean():+.4f} (ICIR {v.mean()/(v.std()+1e-12)*np.sqrt(len(v)):5.2f})', v.mean()

    print('=== 1. 原始 RankIC ===')
    print(f'{"锚":8s} {"dev":>26s} {"val":>26s}')
    for k in ('rev', 'price', 'dual', 'te', 'fin', 'rand'):
        nm = {'rev': '纯反转', 'rand': '随机网络'}.get(k, k)
        print(f'{nm:8s} {f(acc[k]["dev"])[0]:>26s} {f(acc[k]["val"])[0]:>26s}')

    print('\n=== 2. 对纯反转正交化后的残差 IC ===')
    print(f'{"锚":8s} {"dev":>26s} {"val":>26s}')
    for k in ('price', 'dual', 'te', 'fin', 'rand'):
        nm = {'rand': '随机网络'}.get(k, k)
        print(f'{nm:8s} {f(accr[k]["dev"])[0]:>26s} {f(accr[k]["val"])[0]:>26s}')

    print(f'\n=== 3. 随机网络零基准 ({N_RAND} 次) ===')
    out = {'ic': {}, 'resid_ic': {}, 'zero_baseline': {}}
    for s in SEG:
        rm = np.array([np.mean([x for x in r if x is not None and np.isfinite(x)])
                       for r in rand_runs[s] if len(r)])
        fin_m = f(acc['fin'][s])[1]
        z = (fin_m - rm.mean()) / (rm.std() + 1e-12)
        print(f'  {s:4s} 随机网络 IC {rm.mean():+.4f} ± {rm.std():.4f}   '
              f'fin {fin_m:+.4f}   超出 {z:+.1f}σ')
        out['zero_baseline'][s] = {'rand_mean': float(rm.mean()), 'rand_std': float(rm.std()),
                                   'fin': float(fin_m), 'z': float(z)}
    for k in acc:
        out['ic'][k] = {s: f(acc[k][s])[1] for s in SEG}
    for k in accr:
        out['resid_ic'][k] = {s: f(accr[k][s])[1] for s in SEG}
    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump(out, open(f'{OUTPUT_DIR}/research/reversal_control.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\n已写 output/research/reversal_control.json')


if __name__ == '__main__':
    main()
