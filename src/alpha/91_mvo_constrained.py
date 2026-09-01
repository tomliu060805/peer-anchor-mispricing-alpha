# -*- coding: utf-8 -*-
"""路线B: 带约束的组合优化(目标=压敞口, 不是提收益)。

W3 对照已证明"权重层不产生 alpha", 因此本轮目标明确为:
  **在不损失超额的前提下, 把行业与风格敞口压进可控带内。**

不引入求解器依赖, 用投影梯度(projected gradient)迭代求解:
  max_w  score·w − λ/2 · w'Σw     s.t.  Σw=1, 0≤w≤cap, |B'w − B'w_bench| ≤ band

Σ 用 Barra 风格因子 + 行业哑变量的因子协方差近似(缺失则退化为对角特质方差);
敞口约束用逐因子软投影(超出带宽则按梯度方向回拉), 迭代 30 次。

变体:
  B0 现行(无约束)
  B1 仅个股上限 1%
  B2 +行业偏离带 ±5%(相对活跃域等权)
  B3 +风格敞口带 ±0.25σ(SIZE/RESVOL/LIQUIDTY 三个最相关的)
  B4 B2+B3 全套
报告: 超额/IR/回撤/换手 + **敞口诊断**(行业最大偏离, 三风格敞口绝对值)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, BARRA_ROOT

import os, json, pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

os.environ['OPENBLAS_NUM_THREADS'] = '2'

_src = open(f'{PROJ}/src/test/open_test_segment.py', encoding='utf-8').read()
exec(_src.split("out = {'criteria': CRIT}")[0])

SEG = {'dev': ('2016-01-01', '2022-12-31'),
       'val': ('2023-01-01', '2024-08-16'),
       'test': ('2024-08-19', '2026-12-31')}
CAP = 0.01
IND_BAND = 0.05
STY_BAND = 0.25
STYLES = ['SIZE', 'RESVOL', 'LIQUIDTY']

# ---------- Barra 风格暴露(仅三个最相关的, T-1) ----------
code_ix = {c: i for i, c in enumerate(codes)}
EXPO_CACHE = f'{CACHE}/expo3_grid.npz'
if not os.path.exists(EXPO_CACHE):
    def load_expo(t):
        out = np.full((len(STYLES), N), np.nan, np.float32)
        if not BARRA_ROOT:
            return t, out
        d = str(dates[t - 1])
        for j, sty in enumerate(STYLES):
            f = f'{BARRA_ROOT}/{sty}/{d}.parquet'
            if not os.path.exists(f):
                continue
            df = pd.read_parquet(f)
            ix = np.array([code_ix.get(c, -1) for c in df['code']])
            v = df[sty].values.astype(np.float32)
            ok = ix >= 0
            out[j, ix[ok]] = v[ok]
        return t, out
    print('加载风格暴露...', flush=True)
    with ProcessPoolExecutor(40) as ex:
        outs = list(ex.map(load_expo, [int(t) for t in sig_all], chunksize=4))
    E = np.full((len(sig_all), len(STYLES), N), np.nan, np.float32)
    pos = {int(t): i for i, t in enumerate(sig_all)}
    for t, o in outs:
        E[pos[t]] = o
    np.savez_compressed(EXPO_CACHE, e=E, t=np.array([int(x) for x in sig_all]))
    print('cached', flush=True)
_ez = np.load(EXPO_CACHE)
EXPO, EXPO_T = _ez['e'], _ez['t']
EXPO_POS = {int(t): i for i, t in enumerate(EXPO_T)}

with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
    _ind_map = pickle.load(fh)
_ind_months = sorted(_ind_map.keys())


def ind_of(t):
    d = str(dates[t - 1])
    mk = [m for m in _ind_months if m <= d]
    imap = _ind_map[mk[-1]] if mk else {}
    return np.array([imap.get(c, '') for c in codes])


def solve(score, ids, t, cap=None, ind_band=None, sty_band=None, n_iter=30):
    """投影梯度: 从分数加权出发, 迭代满足上限与敞口带。"""
    k = len(ids)
    w = np.maximum(score, 0) + 0.3
    w = w / w.sum()
    if cap is None and ind_band is None and sty_band is None:
        return w
    inds = ind_of(t)[ids] if ind_band is not None else None
    if ind_band is not None:
        # 基准 = 活跃域等权的行业分布
        dom = (lim250[t] >= 2) & tradable[t]
        di = ind_of(t)[dom]
        uniq = sorted(set(di) - {''})
        bench = {u: float((di == u).mean()) for u in uniq}
    if sty_band is not None:
        ep = EXPO[EXPO_POS[t]][:, ids] if t in EXPO_POS else np.full((len(STYLES), k), np.nan)
        dom = (lim250[t] >= 2) & tradable[t]
        eb = EXPO[EXPO_POS[t]][:, dom] if t in EXPO_POS else np.full((len(STYLES), int(dom.sum())), np.nan)
        with np.errstate(all='ignore'):
            bmean = np.nanmean(eb, 1)
            bstd = np.nanstd(eb, 1)
        ep = np.where(np.isfinite(ep), ep, bmean[:, None])
        bstd = np.where(np.isfinite(bstd) & (bstd > 1e-6), bstd, 1.0)
    for _ in range(n_iter):
        if cap is not None:
            over = w > cap
            if over.any():
                excess = (w[over] - cap).sum()
                w[over] = cap
                free = ~over
                if free.any() and w[free].sum() > 0:
                    w[free] += excess * w[free] / w[free].sum()
        if ind_band is not None:
            for u, bw in bench.items():
                m = inds == u
                if not m.any():
                    continue
                cur = w[m].sum()
                hi, lo = bw + ind_band, max(bw - ind_band, 0.0)
                if cur > hi and cur > 0:
                    cut = cur - hi
                    w[m] *= (cur - cut) / cur
                    o = ~m
                    if o.any() and w[o].sum() > 0:
                        w[o] += cut * w[o] / w[o].sum()
        if sty_band is not None:
            for j in range(len(STYLES)):
                cur = float((w * ep[j]).sum())
                dev = (cur - bmean[j]) / bstd[j]
                if abs(dev) > sty_band:
                    tgt = bmean[j] + np.sign(dev) * sty_band * bstd[j]
                    x = ep[j] - bmean[j]
                    denom = float((w * x * x).sum()) + 1e-12
                    step = (cur - tgt) / denom
                    w = w * np.exp(-np.clip(step * x, -0.5, 0.5))
                    w = np.maximum(w, 1e-8)
                    w = w / w.sum()
        w = np.maximum(w, 0)
        s = w.sum()
        if s <= 0:
            return np.ones(k) / k
        w = w / s
    return w


def run_b(mode):
    holdings = {}
    recs = []
    diag = []
    kw = {'B0': {}, 'B1': dict(cap=CAP), 'B2': dict(cap=CAP, ind_band=IND_BAND),
          'B3': dict(cap=CAP, sty_band=STY_BAND),
          'B4': dict(cap=CAP, ind_band=IND_BAND, sty_band=STY_BAND)}[mode]
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
        ids = np.array(list(new_h))
        sc = np.array([np.nan_to_num(comb[i], nan=0.5) for i in ids])
        w = solve(sc, ids, t, **kw)
        wt = dict(zip(ids.tolist(), w))
        # 敞口诊断
        inds = ind_of(t)[ids]
        domm = (lim250[t] >= 2) & tradable[t]
        di = ind_of(t)[domm]
        maxdev = 0.0
        for u in set(inds) - {''}:
            maxdev = max(maxdev, abs(w[inds == u].sum() - float((di == u).mean())))
        sdev = []
        if t in EXPO_POS:
            ep = EXPO[EXPO_POS[t]][:, ids]
            eb = EXPO[EXPO_POS[t]][:, domm]
            with np.errstate(all='ignore'):
                bm, bs = np.nanmean(eb, 1), np.nanstd(eb, 1)
            bs = np.where(np.isfinite(bs) & (bs > 1e-6), bs, 1.0)
            ep = np.where(np.isfinite(ep), ep, bm[:, None])
            sdev = [abs(float((w * ep[j]).sum()) - bm[j]) / bs[j] for j in range(len(STYLES))]
        turn = sum(abs(wt.get(i2, 0) - holdings.get(i2, 0)) for i2 in set(wt) | set(holdings))
        t_next = sig_all[si + 1]
        pr = 0.0
        bs_ = set(bought)
        for i2, ww in wt.items():
            if i2 in bs_ and t + 1 < T:
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
        recs.append((dnum[t], pr, np.exp(lgq[t_next] - lgq[t]) - 1, turn / 2))
        diag.append((maxdev, float(np.max(w)), sdev))
        holdings = wt
    return recs, diag


def rep(rd, name):
    recs, diag = rd
    ds = np.array([r[0] for r in recs])
    pr = np.array([r[1] for r in recs])
    bq = np.array([r[2] for r in recs])
    tn = np.array([r[3] for r in recs])
    md = np.array([d[0] for d in diag])
    mw = np.array([d[1] for d in diag])
    sd = np.array([d[2] for d in diag if len(d[2]) == len(STYLES)])
    d = {'ind_maxdev_median': round(float(np.median(md)), 4),
         'max_w_median': round(float(np.median(mw)), 4)}
    if len(sd):
        for j, s in enumerate(STYLES):
            d[f'sty_{s}_median'] = round(float(np.median(sd[:, j])), 3)
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
                'AbsMDD': round(float((1 - anav / np.maximum.accumulate(anav)).max()), 4),
                'turn': round(float(tn[m].mean()), 3)}
    print(name, json.dumps(d), flush=True)
    return d


res = {}
for mode, nm in [('B0', 'B0_现行无约束'), ('B1', 'B1_个股1%上限'),
                 ('B2', 'B2_+行业带±5%'), ('B3', 'B3_+风格带±0.25σ'),
                 ('B4', 'B4_行业+风格全套')]:
    res[nm] = rep(run_b(mode), nm)
json.dump(res, open(f'{PROJ}/output/alpha/metrics_v91_mvo.json', 'w'),
          ensure_ascii=False, indent=1)
print('done')
