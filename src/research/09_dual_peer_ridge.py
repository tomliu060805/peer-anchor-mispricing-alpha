# -*- coding: utf-8 -*-
"""双重同伴效应的组合层检验: Ridge 拟合 + 多头超额。

三个模型在同一口径下对比:
  base   Ridge 喂 20 个原始特征
  dual   Ridge 喂 40 个特征(每个特征的 peer_avg 与 peer_dev)
  rand   同 dual, 但网络换成随机网络 —— **零基准**

训练: 扩展窗(用截至预测日之前的全部历史), 每 12 周重拟一次。
标的: 未来 5 日收益的截面秩。
评估: 周频 RankIC + 多头 N=200 等权对中证全指的超额(双边 20bp)。

dev/val 报告, test 不碰。dev 段前 2 年用于起步训练故不计入评估。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import numpy as np

MOM, WLIM = 20, 250
NHOLD, COST = 200, 0.0020
REFIT = 12
N_RAND = 5           # 随机零基准的独立抽样次数
SEG = {'dev': ('2018-01-01', '2022-12-31'), 'val': ('2023-01-01', '2024-08-16')}
rng = np.random.default_rng(20260902)


def rk(x, m):
    out = np.full(len(x), np.nan, np.float64)
    ix = np.where(m & np.isfinite(x))[0]
    if len(ix) < 100:
        return out
    r = np.argsort(np.argsort(x[ix])).astype(np.float64)
    out[ix] = (r - r.mean()) / (r.std() + 1e-12)
    return out


def peer_avg_net(X, nbr, wgt):
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


def ridge_fit(X, y, lam=30.0):
    A = np.column_stack([np.ones(len(X)), X])
    P = A.T @ A
    P[1:, 1:] += lam * np.eye(X.shape[1])
    try:
        return np.linalg.solve(P, A.T @ y)
    except np.linalg.LinAlgError:
        return None


def main():
    g = np.load(f'{CACHE}/daily_grid.npz')
    ret, dates, codes, money = g['ret'], g['dates'], g['codes'], g['money']
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
    qz = np.load(f'{CACHE}/quanzhi_ret.npy')
    lgq = np.concatenate([[0.0], np.cumsum(np.log1p(qz[1:]))])

    z = np.load(f'{CACHE}/finratio_grid.npz')
    FR, RN = z['ratios'], [str(x) for x in z['names']]
    EX = {'mom20': mom20, 'vol20': vol20, 'illiq': illiq}
    FEAT = RN + list(EX)
    npz = np.load(f'{CACHE}/nets_price.npz')

    tradable = (paused < 0.5) & (at_hl < 0.5) & (at_ll < 0.5) & (st < 0.5) & ~np.isnan(ret)
    sig = [int(x) for x in np.arange(WLIM + 1, T - 6, 5)]

    # ---- 逐期构造特征 ----
    print('构造特征...', flush=True)
    store = []
    for si, t in enumerate(sig[:-1]):
        t1 = sig[si + 1]
        fwd = np.exp(logc[t1] - logc[t]) - 1
        m = tradable[t] & np.isfinite(fwd)
        if m.sum() < 500:
            continue
        b = np.searchsorted(npz['t'], t, side='right') - 1
        if b < 0:
            continue
        nbr, wgt = npz['n'][b], npz['w'][b]
        pool = np.where(nbr[:, 0] >= 0)[0]
        if len(pool) < 300:
            continue

        rands = []
        for _ in range(N_RAND):
            r_ = np.full_like(nbr, -1)
            r_[pool] = rng.choice(pool, size=(len(pool), nbr.shape[1]))
            rands.append((r_, (r_ >= 0).astype(np.float32)))

        cols = {'base': [], 'dual': []}
        for j in range(N_RAND):
            cols[f'rand{j}'] = []
        for fi, f in enumerate(FEAT):
            X = (FR[t, fi] if fi < len(RN) else EX[f][t]).astype(np.float64)
            xm = m & np.isfinite(X)
            cols['base'].append(rk(X, xm))
            pa = peer_avg_net(X, nbr, wgt)
            cols['dual'] += [rk(pa, xm), rk(X - pa, xm)]
            for j, (r_, w_) in enumerate(rands):
                pa2 = peer_avg_net(X, r_, w_)
                cols[f'rand{j}'] += [rk(pa2, xm), rk(X - pa2, xm)]
        rec = {'t': t, 't1': t1, 'm': m, 'y': rk(fwd, m), 'fwd': fwd}
        for k2, v2 in cols.items():
            rec[k2] = np.column_stack(v2).astype(np.float32)
        store.append(rec)
        if len(store) % 100 == 0:
            print(f'  {len(store)} 期', flush=True)
    print(f'共 {len(store)} 期', flush=True)

    # ---- 扩展窗训练与评估 ----
    res = {}
    MODELS = ['base', 'dual'] + [f'rand{j}' for j in range(N_RAND)]
    for key in MODELS:
        coef = None
        recs = []
        hold = set()
        for i, s in enumerate(store):
            if i % REFIT == 0 and i >= 60:
                Xs, ys = [], []
                for p in store[max(0, i - 400):i]:
                    A, yy = p[key].astype(np.float64), p['y']
                    mm = np.isfinite(yy) & np.isfinite(A).all(1)
                    if mm.sum() > 300:
                        Xs.append(A[mm]); ys.append(yy[mm])
                if Xs:
                    coef = ridge_fit(np.vstack(Xs), np.concatenate(ys))
            if coef is None:
                continue
            A = s[key].astype(np.float64)
            ok = s['m'] & np.isfinite(A).all(1)
            if ok.sum() < 300:
                continue
            pred = np.full(N, np.nan)
            pred[ok] = coef[0] + A[ok] @ coef[1:]
            d = np.datetime64(str(dates[s['t']]))
            seg = next((k for k, (a, b) in SEG.items()
                        if np.datetime64(a) <= d <= np.datetime64(b)), None)
            if seg is None:
                continue
            fwd, yv = s['fwd'], s['y']
            mm = ok & np.isfinite(yv)
            ic = float(np.corrcoef(np.argsort(np.argsort(pred[mm])),
                                   np.argsort(np.argsort(yv[mm])))[0, 1])
            top = np.argsort(-np.nan_to_num(pred, nan=-1e9))[:NHOLD]
            pr = float(np.nanmean(fwd[top]))
            turn = 1.0 - len(hold & set(top.tolist())) / NHOLD
            hold = set(top.tolist())
            bq = float(np.exp(lgq[s['t1']] - lgq[s['t']]) - 1)
            recs.append((seg, ic, pr - turn * COST, bq, turn))
        res[key] = recs

    def stat(recs, s):
        r = [x for x in recs if x[0] == s]
        if len(r) < 20:
            return None
        ic = np.array([x[1] for x in r]); pr = np.array([x[2] for x in r])
        bq = np.array([x[3] for x in r]); tn = np.array([x[4] for x in r])
        ex = (1 + pr) / (1 + bq) - 1
        nav = np.cumprod(1 + ex)
        win, loss = ex[ex > 0], ex[ex < 0]
        return {'ic': float(ic.mean()),
                'icir': float(ic.mean() / (ic.std() + 1e-12) * np.sqrt(len(ic))),
                'ExAnn': float(nav[-1] ** (50.4 / len(ex)) - 1),
                'IR': float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(50.4)),
                'ExMDD': float((1 - nav / np.maximum.accumulate(nav)).max()),
                'n_trade': int(len(ex)), 'winrate': float((ex > 0).mean()),
                'avg_ex': float(ex.mean()),
                'pl_ratio': float(win.mean() / abs(loss.mean())) if len(loss) else np.nan,
                'profit_factor': float(win.sum() / abs(loss.sum())) if len(loss) else np.nan,
                'turn': float(tn.mean())}

    print(f'\n{"模型":10s} {"段":4s} {"RankIC":>8s} {"超额年化":>9s} {"IR":>6s} '
          f'{"MDD":>7s} {"期数":>5s} {"胜率":>6s} {"单期均":>8s} {"盈亏比":>6s} '
          f'{"盈利因子":>7s} {"换手":>6s}')
    out = {}
    for key, nm in (('base', '原始'), ('dual', '双重同伴')):
        out[nm] = {}
        for s in SEG:
            d = stat(res[key], s)
            if not d:
                continue
            out[nm][s] = d
            print(f'{nm:10s} {s:4s} {d["ic"]:8.4f} {d["ExAnn"]:8.2%} {d["IR"]:6.2f} '
                  f'{d["ExMDD"]:6.2%} {d["n_trade"]:5d} {d["winrate"]:5.1%} '
                  f'{d["avg_ex"]:7.3%} {d["pl_ratio"]:6.2f} {d["profit_factor"]:7.2f} '
                  f'{d["turn"]:5.1%}')

    out['随机零基准'] = {}
    print()
    for s in SEG:
        ds = [stat(res[f'rand{j}'], s) for j in range(N_RAND)]
        ds = [d for d in ds if d]
        if not ds:
            continue
        ann = np.array([d['ExAnn'] for d in ds]); ir = np.array([d['IR'] for d in ds])
        dm = out['双重同伴'][s]
        z_ann = (dm['ExAnn'] - ann.mean()) / (ann.std(ddof=1) + 1e-12)
        z_ir = (dm['IR'] - ir.mean()) / (ir.std(ddof=1) + 1e-12)
        out['随机零基准'][s] = {'ExAnn_mean': float(ann.mean()), 'ExAnn_std': float(ann.std(ddof=1)),
                            'IR_mean': float(ir.mean()), 'IR_std': float(ir.std(ddof=1)),
                            'z_ExAnn': float(z_ann), 'z_IR': float(z_ir), 'n_draw': len(ds)}
        print(f'随机零基准({len(ds)}次) {s:4s} 超额 {ann.mean():+.2%} ± {ann.std(ddof=1):.2%}  '
              f'IR {ir.mean():.2f} ± {ir.std(ddof=1):.2f}   '
              f'-> 双重同伴超出 {z_ann:+.1f}σ (超额) / {z_ir:+.1f}σ (IR)')

    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump(out, open(f'{OUTPUT_DIR}/research/dual_peer_ridge.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\n已写 output/research/dual_peer_ridge.json')


if __name__ == '__main__':
    main()
