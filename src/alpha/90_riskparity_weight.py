# -*- coding: utf-8 -*-
"""路线A: 风险平价式权重(不引入优化器, 不新增可调参数)。

动机: 现行权重 ∝ 综合分, 完全不考虑个股风险。200 只活跃小盘彼此高度共动
      (top-5 邻居残差相关中位 0.42), 名义分散但有效自由度低, 是绝对回撤 55% 的结构来源。

变体(全部沿用既有量, 不新增窗口/阈值):
  W0 现行            w ∝ score + 0.3
  W1 风险平价        w ∝ (score + 0.3) / vol20
  W2 风险平价+上限   W1 再加个股上限 1%(=200只等权的2倍), 超出部分按比例回补
  W3 纯逆波动        w ∝ 1 / vol20            (对照: 只看风险不看分数)
  W4 逆残差波动      w ∝ (score+0.3) / resid_vol  (剔市场beta后的特质波动)

分段 dev/val/test 全部报告; test 仅展示, 不用于选择。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ

import os, json
import numpy as np
import pandas as pd

os.environ['OPENBLAS_NUM_THREADS'] = '2'

_src = open(f'{PROJ}/src/test/open_test_segment.py', encoding='utf-8').read()
exec(_src.split("out = {'criteria': CRIT}")[0])

SEG = {'dev': ('2016-01-01', '2022-12-31'),
       'val': ('2023-01-01', '2024-08-16'),
       'test': ('2024-08-19', '2026-12-31')}
CAP = 0.01   # 个股上限 = 等权(0.5%)的两倍

# 特质波动: 剔市场(域内等权)后的 20 日残差波动
print('构建特质波动...', flush=True)
RESVOL = np.full((T, N), np.nan, np.float32)
_r = np.nan_to_num(ret, nan=0.0)
_mkt = np.nanmean(np.where(np.isnan(ret), np.nan, ret), axis=1)
_mkt = np.nan_to_num(_mkt, nan=0.0)
for t in range(20, T):
    X = _r[t - 20:t]
    m = _mkt[t - 20:t]
    denom = float((m * m).sum()) + 1e-12
    beta = (X * m[:, None]).sum(0) / denom
    res = X - m[:, None] * beta[None, :]
    RESVOL[t] = res.std(0)
print('done', flush=True)


def run_w(mode):
    holdings = {}
    recs = []
    for si, t in enumerate(sig_all):
        if si + 1 >= len(sig_all) or t < 8:
            continue
        v = np.maximum(vol20[t] * np.sqrt(20), 1e-4)
        stk = np.stack([rank_xs(PG_b[t] / v), rank_xs(ZS_b[t]), rank_xs(PG_dual[t] / v),
                        rank_xs(PG_te[t] / v), rank_xs(fgap(ROE, t))])
        with np.errstate(all='ignore'):
            base = np.where(np.all(np.isnan(stk), 0), np.nan, np.nanmean(stk, 0))
        dom = (lim250[t] >= 2) & tradable[t] & ~np.isnan(PG_b[t])
        base = np.where(dom, base, np.nan)
        hard = (paused[t] < 0.5) & (at_hl[t] < 0.5) & dom & ~np.isnan(base)
        if hard.sum() < 100:
            holdings = {}
            continue
        fm = dict(zip(FN, tfeat(t)))
        behav_r = np.nanmean(np.stack([rank01(-fm['sweep_sell'], hard),
                                       rank01(-fm['osize_sell'], hard)]), 0)
        sl = [(np.nan_to_num(close_raw[t], nan=0) >= 2.0).astype(np.float32)]
        dr = DROE[t]
        fin = hard & np.isfinite(dr)
        sl.append((np.nan_to_num(dr, nan=0) >= np.nanquantile(dr[fin], 1 / 3)).astype(np.float32)
                  if fin.sum() > 200 else np.ones(N, np.float32))
        struct_r = np.mean(np.stack(sl), 0)
        with np.errstate(all='ignore'):
            comb = 0.4 * rank01(base, hard) + 0.4 * np.nan_to_num(behav_r, nan=0.5) + 0.2 * struct_r
        comb = np.where(hard, comb, np.nan)
        order = np.argsort(-np.nan_to_num(comb, nan=-1e9))
        rank = np.full(N, 1 << 30)
        rank[order] = np.arange(N)
        can_sell = (paused[t] < 0.5) & (at_ll[t] < 0.5)
        buy_ok = np.ones(N, bool)
        sell_ok = np.ones(N, bool)
        if t + 1 < T:
            c0 = close_raw[t]
            o1 = C30[t + 1, 0]
            with np.errstate(all='ignore'):
                ovr = o1 / c0 - 1
            buy_ok = (paused[t + 1] < 0.5) & (np.nan_to_num(ovr, nan=9) < 0.095)
            sell_ok = (paused[t + 1] < 0.5) & (np.nan_to_num(ovr, nan=-9) > -0.095)
        new_h = dict(holdings)
        sold = []
        for i2 in list(new_h):
            if ((rank[i2] >= 600) or np.isnan(comb[i2])) and can_sell[i2] and sell_ok[i2]:
                sold.append(i2)
                del new_h[i2]
        bought = []
        for i2 in order:
            if len(new_h) >= 200:
                break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2] or not buy_ok[i2]:
                continue
            new_h[i2] = 0.0
            bought.append(i2)
        if len(new_h) < 40:
            holdings = {}
            continue
        ids = list(new_h)
        sc = np.array([np.nan_to_num(comb[i], nan=0.5) + 0.3 for i in ids])
        vl = np.array([vol20[t, i] for i in ids])
        vl = np.where(np.isfinite(vl) & (vl > 1e-6), vl, np.nanmedian(vl[np.isfinite(vl)]) if np.isfinite(vl).any() else 0.02)
        rv = np.array([RESVOL[t, i] for i in ids])
        rv = np.where(np.isfinite(rv) & (rv > 1e-6), rv, np.nanmedian(rv[np.isfinite(rv)]) if np.isfinite(rv).any() else 0.02)
        if mode == 'W0':
            w = sc
        elif mode == 'W1':
            w = sc / vl
        elif mode == 'W2':
            w = sc / vl
            w = w / w.sum()
            for _ in range(20):                      # 迭代封顶并按比例回补
                over = w > CAP
                if not over.any():
                    break
                excess = (w[over] - CAP).sum()
                w[over] = CAP
                free = ~over
                if not free.any() or w[free].sum() <= 0:
                    break
                w[free] += excess * w[free] / w[free].sum()
        elif mode == 'W3':
            w = 1.0 / vl
        else:                                        # W4
            w = sc / rv
        w = w / w.sum()
        wt = dict(zip(ids, w))
        turn = sum(abs(wt.get(i2, 0) - holdings.get(i2, 0)) for i2 in set(wt) | set(holdings))
        t_next = sig_all[si + 1]
        pr = 0.0
        bs = set(bought)
        for i2, ww in wt.items():
            if i2 in bs and t + 1 < T:
                p_in = C30[t + 1, 4, i2]
                c1 = close_raw[t + 1, i2]
                c0 = close_raw[t, i2]
                if (p_in == p_in and c1 == c1 and c0 == c0 and c0 > 0
                        and abs(C30[t + 1, 0, i2] / c0 - 1) <= 0.15):
                    r = (c1 / p_in) * np.exp(logc[t_next, i2] - logc[t + 1, i2]) - 1
                else:
                    r = np.exp(logc[t_next, i2] - logc[t, i2]) - 1
            else:
                r = np.exp(logc[t_next, i2] - logc[t, i2]) - 1
            pr += ww * r
        for i2 in sold:
            c0 = close_raw[t, i2]
            po_ = C30[t + 1, 3, i2] if t + 1 < T else np.nan
            if po_ == po_ and c0 == c0 and c0 > 0 and abs(C30[t + 1, 0, i2] / c0 - 1) <= 0.15:
                pr += holdings.get(i2, 0) * (po_ / c0 - 1)
        pr -= turn * COST
        recs.append((dnum[t], pr, np.exp(lgq[t_next] - lgq[t]) - 1, turn / 2, float(np.max(w))))
        holdings = wt
    return recs


def rep(recs, name):
    ds = np.array([r[0] for r in recs])
    pr = np.array([r[1] for r in recs])
    bq = np.array([r[2] for r in recs])
    tn = np.array([r[3] for r in recs])
    mx = np.array([r[4] for r in recs])
    d = {'max_w_median': round(float(np.median(mx)), 4)}
    for k, (a, b) in SEG.items():
        m = (ds >= np.datetime64(a)) & (ds <= np.datetime64(b))
        if m.sum() < 10:
            continue
        ex = (1 + pr[m]) / (1 + bq[m]) - 1
        nav = np.cumprod(1 + ex)
        anav = np.cumprod(1 + pr[m])
        d[k] = {'ExAnn': round(float(nav[-1] ** (50.4 / len(ex)) - 1), 4),
                'IR': round(float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(50.4)), 2),
                'ExMDD': round(float((1 - nav / np.maximum.accumulate(nav)).max()), 4),
                'AbsVol': round(float(pr[m].std() * np.sqrt(50.4)), 4),
                'AbsMDD': round(float((1 - anav / np.maximum.accumulate(anav)).max()), 4),
                'turn': round(float(tn[m].mean()), 3)}
    print(name, json.dumps(d), flush=True)
    return d


res = {}
for mode, nm in [('W0', 'W0_现行(∝分数)'), ('W1', 'W1_风险平价(分数/vol)'),
                 ('W2', 'W2_风险平价+1%上限'), ('W3', 'W3_纯逆波动(对照)'),
                 ('W4', 'W4_逆特质波动')]:
    res[nm] = rep(run_w(mode), nm)
json.dump(res, open(f'{PROJ}/output/alpha/metrics_v90_riskparity.json', 'w'),
          ensure_ascii=False, indent=1)
print('done')
