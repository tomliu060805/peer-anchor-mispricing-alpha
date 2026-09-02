# -*- coding: utf-8 -*-
"""双重同伴效应: 把特征拆成"同伴均值"与"个股偏离", 检验两者系数是否分化。

思想
----
标准截面回归 r = b·X 隐含一个未被检验的约束: 因为 X = peer_avg + peer_dev,
写成 r = b·peer_avg + b·peer_dev 时, 强制两项同系数。若同伴均值与个体偏离
的预测方向本就不同(如同伴整体估值过高为负、个股相对同伴低估为正), 拆开各给
一个系数能拿到更精确的信息。

与本项目的关系
--------------
本项目的核心锚 gap = 邻居动量加权均值 − 自身动量, 而 peer_dev = X − peer_avg,
所以 **gap 恰好等于 −peer_dev**。即我们只用了偏离项, 从未把同伴均值当独立信号,
等于把约束设在了另一端(隐含 peer_avg 系数为 0)。本脚本检验该约束是否成立。

必须的零基准
------------
peer_avg 在随机网络上退化为全域均值的噪声估计。若真实网络的 peer_avg 项
不优于随机网络的, 那它就与"同伴"无关。随机网络零基准与真实网络同规模同口径,
跑 N_RAND 次取分布。判据见 [[random-baseline-for-gap-anchors]] 的教训。

分段: dev 2016-2022 / val 2023~2024-08-16 报告, test 不碰。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np

MOM, WLIM = 20, 250
SEG = {'dev': ('2016-01-01', '2022-12-31'), 'val': ('2023-01-01', '2024-08-16')}
N_RAND = 10
rng = np.random.default_rng(20260902)


def rk(x, m):
    """截面秩标准化(仅在 m 上), 其余置 NaN。"""
    out = np.full(len(x), np.nan, np.float64)
    ix = np.where(m & np.isfinite(x))[0]
    if len(ix) < 100:
        return out
    r = np.argsort(np.argsort(x[ix])).astype(np.float64)
    out[ix] = (r - r.mean()) / (r.std() + 1e-12)
    return out


def peer_avg_net(X, nbr, wgt):
    """按网络算同伴均值(不含自身——网络本就不含自环)。"""
    N = len(X)
    out = np.full(N, np.nan, np.float64)
    idx = np.where((nbr[:, 0] >= 0) & (wgt.sum(1) > 1e-12))[0]
    if len(idx) < 100:
        return out
    nb, w0 = nbr[idx], np.maximum(wgt[idx], 0)
    pv = X[np.where(nb >= 0, nb, 0)]
    msk = (nb >= 0) & np.isfinite(pv)
    w = w0 * msk
    ok = w.sum(1) > 1e-12
    out[idx[ok]] = ((np.nan_to_num(pv) * w).sum(1) / np.maximum(w.sum(1), 1e-12))[ok]
    return out


def peer_avg_group(X, grp, valid):
    """按分组(行业)算留一均值: (组和 − 自身) / (组数 − 1)。"""
    N = len(X)
    out = np.full(N, np.nan, np.float64)
    ok = valid & np.isfinite(X)
    for u in np.unique(grp[ok]):
        if u == '':
            continue
        m = ok & (grp == u)
        c = int(m.sum())
        if c < 5:
            continue
        s = X[m].sum()
        out[m] = (s - X[m]) / (c - 1)
    return out


def fm_reg(y, cols):
    """截面 OLS, 返回系数向量(含截距)。cols 为回归元列表。"""
    m = np.isfinite(y)
    for c in cols:
        m &= np.isfinite(c)
    if m.sum() < 200:
        return None
    A = np.column_stack([np.ones(m.sum())] + [c[m] for c in cols])
    try:
        b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    except np.linalg.LinAlgError:
        return None
    return b[1:]


def main():
    g = np.load(f'{CACHE}/daily_grid.npz')
    ret, dates, codes = g['ret'], g['dates'], g['codes']
    money = g['money']
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
    with np.errstate(all='ignore'):
        il = np.abs(r0) / np.maximum(money, 1.0) * 1e8
    ci = np.cumsum(np.nan_to_num(il, nan=0.0), 0)
    illiq = np.full((T, N), np.nan, np.float32)
    illiq[MOM:] = ((ci[MOM:] - ci[:-MOM]) / MOM).astype(np.float32)

    z = np.load(f'{CACHE}/finratio_grid.npz')
    FR, RN = z['ratios'], [str(x) for x in z['names']]
    FEAT = RN + ['mom20', 'vol20', 'illiq']

    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    months = sorted(im)

    nets = {'K5': np.load(f'{CACHE}/nets_price.npz')}
    v17 = np.load(f'{CACHE}/nets_v17.npz')
    nets['K20'] = {'t': v17['rb21'], 'n': v17['n21_20'], 'w': v17['w21_20']}

    tradable = (paused < 0.5) & (at_hl < 0.5) & (at_ll < 0.5) & (st < 0.5) & ~np.isnan(ret)
    sig = [int(x) for x in np.arange(WLIM + 1, T - 6, 5)]

    NETS = ['K5', 'K20', 'IND', 'R5', 'R20']
    acc = {s: {n: {f: {'avg': [], 'dev': []} for f in FEAT} for n in NETS} for s in SEG}
    acc_raw = {s: {f: [] for f in FEAT} for s in SEG}

    for si, t in enumerate(sig[:-1]):
        d = np.datetime64(str(dates[t]))
        seg = next((s for s, (a, b) in SEG.items()
                    if np.datetime64(a) <= d <= np.datetime64(b)), None)
        if seg is None:
            continue
        t1 = sig[si + 1]
        fwd = np.exp(logc[t1] - logc[t]) - 1
        base_m = tradable[t] & np.isfinite(fwd)
        if base_m.sum() < 500:
            continue
        y = rk(fwd, base_m)

        mk = [m for m in months if m <= str(dates[t - 1])]
        inds = np.array([im[mk[-1]].get(c, '') for c in codes]) if mk else np.full(N, '')

        nb = {}
        for k in ('K5', 'K20'):
            zz = nets[k]
            b = np.searchsorted(zz['t'], t, side='right') - 1
            nb[k] = (zz['n'][b], zz['w'][b]) if b >= 0 else None
        for k, src in (('R5', 'K5'), ('R20', 'K20')):
            if nb[src] is None:
                nb[k] = None
                continue
            n0, w0 = nb[src]
            pool = np.where(n0[:, 0] >= 0)[0]
            Kk = n0.shape[1]
            rn = np.full_like(n0, -1)
            rn[pool] = rng.choice(pool, size=(len(pool), Kk))
            nb[k] = (rn, (rn >= 0).astype(np.float32))

        for fi, f in enumerate(FEAT):
            X = (FR[t, fi] if fi < len(RN) else
                 {'mom20': mom20, 'vol20': vol20, 'illiq': illiq}[f][t]).astype(np.float64)
            xm = base_m & np.isfinite(X)
            if xm.sum() < 500:
                continue
            xr = rk(X, xm)
            b = fm_reg(y, [xr])
            if b is not None:
                acc_raw[seg][f].append(float(b[0]))
            for k in NETS:
                if k == 'IND':
                    pa = peer_avg_group(X, inds, xm)
                else:
                    if nb[k] is None:
                        continue
                    pa = peer_avg_net(X, nb[k][0], nb[k][1])
                pd_ = X - pa
                par, pdr = rk(pa, xm), rk(pd_, xm)
                bb = fm_reg(y, [par, pdr])
                if bb is not None:
                    acc[seg][k][f]['avg'].append(float(bb[0]))
                    acc[seg][k][f]['dev'].append(float(bb[1]))

    def tstat(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            return np.nan, np.nan
        return float(v.mean()), float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))

    out = {'feat': FEAT, 'seg': {}}
    for s in SEG:
        print(f'\n{"="*86}\n段 {s}\n{"="*86}')
        raw_t, raw_b = [], []
        for f in FEAT:
            b_, t_ = tstat(acc_raw[s][f])
            if np.isfinite(t_):
                raw_t.append(abs(t_))
                raw_b.append(abs(b_))
        print(f'原始因子       |T| 均值 {np.mean(raw_t):6.2f}   |系数| 均值 {np.mean(raw_b):.5f}   '
              f'({len(raw_t)} 个特征)')
        rec = {'raw': {'mean_abs_t': float(np.mean(raw_t)),
                       'mean_abs_b': float(np.mean(raw_b))}, 'nets': {}}
        print(f'\n{"网络":6s} {"peer_avg |T|":>13s} {"peer_dev |T|":>13s} '
              f'{"avg|系数|":>10s} {"dev|系数|":>10s} {"系数异号占比":>12s}')
        for k in NETS:
            at, dt, ab, db, opp = [], [], [], [], 0
            n = 0
            for f in FEAT:
                ba, ta = tstat(acc[s][k][f]['avg'])
                bd, td = tstat(acc[s][k][f]['dev'])
                if not (np.isfinite(ta) and np.isfinite(td)):
                    continue
                n += 1
                at.append(abs(ta)); dt.append(abs(td))
                ab.append(abs(ba)); db.append(abs(bd))
                opp += (ba * bd < 0)
            if not n:
                continue
            nm = {'R5': '随机K5', 'R20': '随机K20', 'IND': '行业'}.get(k, k)
            print(f'{nm:6s} {np.mean(at):13.2f} {np.mean(dt):13.2f} '
                  f'{np.mean(ab):10.5f} {np.mean(db):10.5f} {opp/n:11.1%}')
            rec['nets'][k] = {'avg_abs_t': float(np.mean(at)), 'dev_abs_t': float(np.mean(dt)),
                              'avg_abs_b': float(np.mean(ab)), 'dev_abs_b': float(np.mean(db)),
                              'opposite_sign_share': float(opp / n), 'n_feat': n}
        out['seg'][s] = rec

    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump(out, open(f'{OUTPUT_DIR}/research/dual_peer.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\n已写 output/research/dual_peer.json')


if __name__ == '__main__':
    main()
