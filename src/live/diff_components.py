# -*- coding: utf-8 -*-
"""逐块比较生产打分与回测打分, 定位偏差来源。

reconcile.py 只告诉你"不一致", 这个脚本告诉你"哪一块不一致"。
对每个中间量报告: 两侧非空数量、共同非空处的相关系数与最大绝对差。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np

_src = open(f'{PROJ}/src/test/open_test_segment.py', encoding='utf-8').read()
exec(_src.split("out = {'criteria': CRIT}")[0])

_sys.path.insert(0, f'{PROJ}/src/live')
import generate_portfolio as gp

# 只取回测信号网格上的日期 —— 非网格日期上回测侧的锚变量是 NaN/陈旧值
DATES = [str(dates[t]) for t in sig_all[-25::12]]


def cmp(name, a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ma, mb = np.isfinite(a), np.isfinite(b)
    both = ma & mb
    if both.sum() < 10:
        print(f'  {name:14s} 非空 {ma.sum():5d}/{mb.sum():5d}  共同非空太少({both.sum()})')
        return
    c = np.corrcoef(a[both], b[both])[0, 1]
    mx = np.abs(a[both] - b[both]).max()
    flag = '' if (c > 0.999 and mx < 1e-4) else '   <<< 不一致'
    print(f'  {name:14s} 非空 {ma.sum():5d}/{mb.sum():5d}  共同 {both.sum():5d}  '
          f'相关 {c:+.6f}  最大差 {mx:.3e}{flag}')


for ds in DATES:
    w = np.where(dates == ds)[0]
    if not len(w):
        continue
    t = int(w[0])
    print(f'\n===== {ds} (t={t}) =====')

    _, _, _, ex = gp.generate(asof=ds, write=False, return_scores=True)
    P = ex['dbg']

    v = np.maximum(vol20[t] * np.sqrt(20), 1e-4)
    cmp('vol20*sqrt20', P['v'], v)
    cmp('mom20', P['mom20'], mom20[t])
    cmp('gap_price', P['g_p'], PG_b[t])
    cmp('zspread', P['zs'], ZS_b[t])
    cmp('gap_dual', P['g_d'], PG_dual[t])
    cmp('gap_te', P['g_t'], PG_te[t])
    cmp('gap_roe', P['g_roe'], fgap(ROE, t))
    cmp('droe', P['droe'], DROE[t])

    dom_b = (lim250[t] >= 2) & tradable[t] & ~np.isnan(PG_b[t])
    print(f'  域大小 生产 {int(P["dom"].sum())} / 回测 {int(dom_b.sum())}  '
          f'交集 {int((P["dom"] & dom_b).sum())}')
    print(f'  lim250>=2 生产 {int((P["lim250"]>=2).sum())} / 回测 {int((lim250[t]>=2).sum())}')
    print(f'  tradable  生产 {int(P["tradable"].sum())} / 回测 {int(tradable[t].sum())}')

    stk = np.stack([rank_xs(PG_b[t] / v), rank_xs(ZS_b[t]), rank_xs(PG_dual[t] / v),
                    rank_xs(PG_te[t] / v), rank_xs(fgap(ROE, t))])
    with np.errstate(all='ignore'):
        base_b = np.where(np.all(np.isnan(stk), 0), np.nan, np.nanmean(stk, 0))
    base_b = np.where(dom_b, base_b, np.nan)
    hard_b = (paused[t] < 0.5) & (at_hl[t] < 0.5) & dom_b & ~np.isnan(base_b)
    cmp('base_score', P['base_s'], base_b)

    fm = dict(zip(FN, tfeat(t)))
    cmp('sweep_sell', P['sweep'], fm['sweep_sell'])
    cmp('osize_sell', P['osize'], fm['osize_sell'])
    behav_b = np.nanmean(np.stack([rank01(-fm['sweep_sell'], hard_b),
                                   rank01(-fm['osize_sell'], hard_b)]), 0)
    cmp('behavior', P['behav'], behav_b)

    sl = [(np.nan_to_num(close_raw[t], nan=0) >= 2.0).astype(np.float32)]
    dr = DROE[t]; fin = hard_b & np.isfinite(dr)
    sl.append((np.nan_to_num(dr, nan=0) >= np.nanquantile(dr[fin], 1 / 3)).astype(np.float32)
              if fin.sum() > 200 else np.ones(N, np.float32))
    cmp('structure', P['struct'], np.mean(np.stack(sl), 0))

    with np.errstate(all='ignore'):
        comb_b = 0.4 * rank01(base_b, hard_b) + 0.4 * np.nan_to_num(behav_b, nan=0.5) + 0.2 * np.mean(np.stack(sl), 0)
    comb_b = np.where(hard_b, comb_b, np.nan)
    cmp('COMB', ex['comb'], comb_b)
