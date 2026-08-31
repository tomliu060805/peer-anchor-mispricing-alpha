# -*- coding: utf-8 -*-
"""测试段一次性开封 (2024-08-19 起)。

事前判定标准 (docs/open_issues.md, 开封前写定, 开封后不得调整):
    超额 > 0 且 IR > 0.5   -> 通过
    超额 > 0 但 IR <= 0.5  -> 存疑, 仅小资金试盘
    超额 <= 0              -> 判负, 整条线终止

配置完全取自 v27 定型 + 融合 h=1.0, 不做任何参数调整。本脚本只运行一次。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, hashlib
import numpy as np
import pandas as pd

os.environ['OPENBLAS_NUM_THREADS'] = '2'
R = PROJ+''

# 复用 v27 定型脚本的数据装载与信号构建段
_src = open(f'{R}/src/alpha/86_lean_v27.py', encoding='utf-8').read()
exec(_src.split("\nres={}\n")[0])

TEST0 = np.datetime64('2024-08-19')

CRIT = {
    'pre_registered': 'docs/open_issues.md',
    'unseal_date': '2026-08-28',
    'rule': {'pass': 'ExAnn>0 and IR>0.5',
             'doubt': 'ExAnn>0 and IR<=0.5',
             'fail': 'ExAnn<=0'},
    'config': 'v27(五锚40%+微观行为40%+结构20%, N200, 缓冲带600, 周调) + 融合 h=1.0 (CNI2000破位空头)',
    'cost': '双边20bp', 'benchmark': '中证全指000985', 'test_start': '2024-08-19',
}
_js = json.dumps(CRIT, ensure_ascii=False, indent=1)
print('=' * 100)
print('事前判定标准 SHA:', hashlib.sha256(_js.encode()).hexdigest()[:12])
print(_js, flush=True)


# ==================== 为 test 段补算信号 ====================
# 根脚本的 sig5 被 END=2024-08-16 截断, 此处按同一公式为扩展日期补算,
# 不改动任何参数, 仅把计算范围延伸到数据末端。
sig_all = [int(x) for x in np.arange(int(RB8[0]) + 1, T - 6, 5)]
_new = [t for t in sig_all if t not in set(sig5)]
print(f'\n补算信号: 原 {len(sig5)} 期 -> 全 {len(sig_all)} 期, 新增 {len(_new)} 期', flush=True)

_nz41 = np.load(f'{CACHE}/nets_v41_flow.npz')
_RBd, _NBd, _WVd = _nz41['dual_t'], _nz41['dual_n'], _nz41['dual_w']
_nz47 = np.load(f'{CACHE}/nets_v47_te.npz')
_RBt, _NBt, _WVt = _nz47['t'], _nz47['n'], _nz47['w']

def _gap_from(RBx, NB, WV, t):
    b = np.searchsorted(RBx, t, side='right') - 1
    if b < 0:
        return np.full(N, np.nan, np.float32), np.full(N, np.nan, np.float32)
    nbr = NB[b]; wgt = np.maximum(WV[b], 0)
    idx = np.where((nbr[:, 0] >= 0) & (wgt.sum(1) > 1e-12))[0]
    g = np.full(N, np.nan, np.float32); pmo = np.full(N, np.nan, np.float32)
    if len(idx) < 100:
        return g, pmo
    nb = nbr[idx]; w0 = wgt[idx]
    m = mom20[t]; pm = m[np.where(nb >= 0, nb, 0)]
    msk = (nb >= 0) & ~np.isnan(pm); w = w0 * msk
    sw = np.maximum(w.sum(1), 1e-12)
    mu = (np.nan_to_num(pm) * w).sum(1) / sw
    ok = w.sum(1) > 1e-12
    g[idx[ok]] = mu[ok] - m[idx[ok]]; pmo[idx[ok]] = mu[ok]
    return g, pmo

for _t in _new:
    _g, _pm = _gap_from(RB8, NB8, WV8, _t)
    PG_b[_t] = _g; PM_b[_t] = _pm
    PG_dual[_t] = _gap_from(_RBd, _NBd, _WVd, _t)[0]
    PG_te[_t] = _gap_from(_RBt, _NBt, _WVt, _t)[0]

# zspread: 按 rebuild 窗口固化训练统计量后, 对窗口内的新日期求 z
_W = 120
for _b in range(len(RB8)):
    _t1 = int(RB8[_b]); _t0 = _t1 - _W
    _t2 = int(RB8[_b + 1]) if _b + 1 < len(RB8) else T
    _sds = [t for t in _new if _t1 <= t < _t2 and t < T]
    if not _sds:
        continue
    _nbr = NB8[_b]; _idx = np.where(_nbr[:, 0] >= 0)[0]
    if len(_idx) < 100:
        continue
    _nb = _nbr[_idx]
    _base = logc[_t0 - 1] if _t0 > 0 else np.zeros(N)
    _Pw = np.exp(logc[_t0:_t1] - _base)
    _nbc = np.where(_nb >= 0, _nb, 0)
    _sj = _Pw[:, _nbc.ravel()].reshape(_W, len(_idx), _nb.shape[1])
    _str = _Pw[:, _idx][:, :, None] - _sj
    _mu = _str.mean(0); _sd = _str.std(0) + 1e-9
    _d = (_str ** 2).sum(0); _d = _d - _d.min(1, keepdims=True)
    _wsm = np.exp(-_d) * (_nb >= 0)
    _wsm = _wsm / np.maximum(_wsm.sum(1, keepdims=True), 1e-12)
    for _t in _sds:
        _Pt = np.exp(logc[_t] - _base)
        _zv = (_Pt[_idx][:, None] - _Pt[_nbc] - _mu) / _sd
        ZS_b[_t, _idx] = -np.nansum(_zv * _wsm, 1)
print('信号补算完成', flush=True)
# ============================================================


def run_alpha(daily=False):
    """v27 配置, 路径B(T+1 拆分执行)。daily=True 时返回日频收益序列。"""
    holdings = {}
    recs = []
    dly = {}
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
        w = np.array([np.nan_to_num(comb[i2], nan=0.5) + 0.3 for i2 in new_h])
        w = w / w.sum()
        wt = dict(zip(new_h, w))
        turn = sum(abs(wt.get(i2, 0) - holdings.get(i2, 0)) for i2 in set(wt) | set(holdings))
        t_next = sig_all[si + 1]
        if daily:
            ids = np.array(list(wt.keys()))
            cur = np.array([wt[i] for i in ids])
            for d in range(t + 1, min(t_next, T - 1) + 1):
                r = np.nan_to_num(ret[d, ids], nan=0.0)
                p = float((cur * r).sum() / max(cur.sum(), 1e-12))
                if d == t + 1:
                    p -= turn * COST
                dly[d] = dly.get(d, 0.0) + p
                cur = cur * (1 + r)
            holdings = wt
            continue
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
        recs.append((dnum[t], pr, np.exp(lgq[t_next] - lgq[t]) - 1, turn / 2))
        holdings = wt
    if daily:
        ks = np.array(sorted(dly.keys()))
        return ks, np.array([dly[k] for k in ks])
    return recs


def report(pr, bq, name, per_year):
    ex = (1 + pr) / (1 + bq) - 1
    nav = np.cumprod(1 + ex)
    anav = np.cumprod(1 + pr)
    d = {'n': int(len(ex)),
         'ExAnn': round(float(nav[-1] ** (per_year / len(ex)) - 1), 4),
         'IR': round(float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(per_year)), 2),
         'ExMDD': round(float((1 - nav / np.maximum.accumulate(nav)).max()), 4),
         'win': round(float((ex > 0).mean()), 3),
         'avg_bp': round(float(ex.mean() * 1e4), 1),
         'AbsAnn': round(float(anav[-1] ** (per_year / len(pr)) - 1), 4),
         'AbsMDD': round(float((1 - anav / np.maximum.accumulate(anav)).max()), 4)}
    print(f'{name}: {json.dumps(d)}', flush=True)
    return d


def beta_daily_pnl(dd, code='CNI2000', K=2.5, NDAY=14, FEE=10e-4,
                   DEC=(1000, 1029, 1129, 1359), UNIT=0.25):
    """噪声区间破位空头腿, 日频 P&L(记在平仓日)。"""
    df = pd.read_parquet(f'{R}/data/idx1m.parquet')
    gg = df[df['code'] == code]
    pc = gg.pivot_table(index='date', columns='hm', values='close')
    po = gg.pivot_table(index='date', columns='hm', values='open')
    hms = np.array(sorted(pc.columns))
    hp = {h: i for i, h in enumerate(hms)}
    dopen = po[hms[0]]
    sg = {Td: (pc[Td] / dopen - 1).abs().rolling(NDAY, min_periods=NDAY).mean().shift(1)
          for Td in DEC}
    C, O = pc.values, po.values
    nxt = dopen.shift(-1).values
    dts = pd.to_datetime(pc.index)
    pnl = {}
    for di in range(len(pc.index) - 1):
        o0 = dopen.iloc[di]
        if not np.isfinite(o0):
            continue
        for Td in DEC:
            s = sg[Td].iloc[di]
            if not np.isfinite(s):
                continue
            j = hp.get(Td)
            if j is None or j + 1 >= len(hms):
                continue
            if not np.isfinite(C[di, j]) or C[di, j] >= o0 * (1 - K * s):
                continue
            e, x = O[di, j + 1], nxt[di]
            if not (np.isfinite(e) and np.isfinite(x)):
                continue
            pnl[dts[di + 1]] = pnl.get(dts[di + 1], 0.0) + UNIT * ((e / x - 1) - FEE * 2)
    out_ = np.zeros(len(dd))
    mp = {pd.Timestamp(z): i for i, z in enumerate(dd)}
    for k, v in pnl.items():
        i = mp.get(pd.Timestamp(k))
        if i is not None:
            out_[i] = v
    return out_


out = {'criteria': CRIT}

# ---- 周频产品口径 ----
recs = run_alpha()
ds = np.array([r[0] for r in recs])
pr = np.array([r[1] for r in recs])
bq = np.array([r[2] for r in recs])
m = ds >= TEST0
print('\n' + '=' * 100)
print(f'TEST 段: {ds[m][0]} ~ {ds[m][-1]}   ({int(m.sum())} 周)', flush=True)
out['alpha_weekly_test'] = report(pr[m], bq[m], 'alpha周频(路径B) TEST', 252 / 5)
mv = (ds >= np.datetime64('2023-01-01')) & (ds <= np.datetime64('2024-08-16'))
out['alpha_weekly_val_ref'] = report(pr[mv], bq[mv], 'alpha周频 VAL(对照)', 252 / 5)

# ---- 日频 + 融合 ----
dk, dpr = run_alpha(daily=True)
dd = dnum[dk]
qz = np.load(f'{CACHE}/quanzhi_ret.npy')
dbq = qz[dk]
bd = beta_daily_pnl(dd)
mt = dd >= TEST0
out['alpha_daily_test'] = report(dpr[mt], dbq[mt], 'alpha日频 TEST', 252)
out['beta_test'] = report(bd[mt], np.zeros(int(mt.sum())), 'beta单腿 TEST(绝对口径)', 252)
out['fusion_test'] = report((dpr + bd)[mt], dbq[mt], '融合h=1.0 TEST', 252)


def verdict(d):
    if d['ExAnn'] <= 0:
        return 'FAIL 判负'
    return 'PASS 通过' if d['IR'] > 0.5 else 'DOUBT 存疑'


print('\n' + '=' * 100 + '\n判定:')
for k in ['alpha_weekly_test', 'fusion_test']:
    out[k]['verdict'] = verdict(out[k])
    print(f"  {k}: ExAnn={out[k]['ExAnn']:.2%}  IR={out[k]['IR']:.2f}  ->  {out[k]['verdict']}")

np.savez(f'{R}/output/test/test_series.npz',
         dates=dd[mt].astype(str), alpha=dpr[mt], beta=bd[mt], bench=dbq[mt])
json.dump(out, open(f'{R}/output/test/test_unsealed.json', 'w'), ensure_ascii=False, indent=1)
print('\nsaved output/test/test_unsealed.json')
