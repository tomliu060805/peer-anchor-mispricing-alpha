# -*- coding: utf-8 -*-
"""S6: 当期打分与持仓建议 (v28 配置)。

产出
----
  output/live/portfolio_{date}.csv    当期目标持仓 (代码/权重/分数/各块得分)
  output/live/orders_{date}.csv       相对上期持仓的买卖指令
  output/live/log.jsonl               append-only 决策日志(含配置SHA, 事后不可改)

配置来源: config/frozen_alpha_v28.json —— 脚本读取该文件并校验 SHA,
配置变更必须先改 json 再跑, 避免"代码里偷偷调参"。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, CACHE_DIR as CACHE, OUTPUT_DIR, CONFIG_DIR

import os, json, time, pickle, hashlib
import numpy as np
import pandas as pd

W, K, MOM, WLIM = 120, 5, 20, 250
N_HOLD, BUFFER_MULT = 200, 3.0
CAP, IND_BAND = 0.01, 0.05
WB, WH, WS = 0.4, 0.4, 0.2


def _rank_xs(x):
    m = ~np.isnan(x)
    r = np.full_like(x, np.nan, dtype=np.float32)
    if m.sum() < 50:
        return r
    rr = np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m] = (rr - rr.mean()) / (rr.std() + 1e-12)
    return r


def _rank01(x, mask):
    out = np.full(len(x), np.nan, np.float32)
    ix = np.where(mask & ~np.isnan(x))[0]
    if len(ix) < 10:
        return out
    r = np.argsort(np.argsort(x[ix])).astype(np.float32) / max(len(ix) - 1, 1)
    out[ix] = r
    return out


def _gap(NBz, t, mom):
    """由某张网络算 (邻居动量加权均值 − 自身)。"""
    RB, NB, WV = NBz['t'], NBz['n'], NBz['w']
    b = np.searchsorted(RB, t, side='right') - 1
    N = NB.shape[1]
    g = np.full(N, np.nan, np.float32)
    pm_out = np.full(N, np.nan, np.float32)
    if b < 0:
        return g, pm_out
    nbr, wgt = NB[b], np.maximum(WV[b], 0)
    idx = np.where((nbr[:, 0] >= 0) & (wgt.sum(1) > 1e-12))[0]
    if len(idx) < 100:
        return g, pm_out
    nb, w0 = nbr[idx], wgt[idx]
    pm = mom[np.where(nb >= 0, nb, 0)]
    msk = (nb >= 0) & ~np.isnan(pm)
    w = w0 * msk
    sw = np.maximum(w.sum(1), 1e-12)
    mu = (np.nan_to_num(pm) * w).sum(1) / sw
    ok = w.sum(1) > 1e-12
    g[idx[ok]] = mu[ok] - mom[idx[ok]]
    pm_out[idx[ok]] = mu[ok]
    return g, pm_out


def generate(asof=None, write=True, return_scores=False):
    cfg_path = f'{CONFIG_DIR}/frozen_alpha_v28.json'
    cfg = json.loads(open(cfg_path, encoding='utf-8').read())
    # 只对**参数段**取哈希: evidence / known_limits 是文档, 补一条已知局限
    # 不该让"同 SHA = 同配置"这个不变量失效, 否则历史决策日志无法按 SHA 归组。
    PARAM_KEYS = ['version', 'domain', 'score', 'portfolio', 'execution', 'cost',
                  'benchmark', 'split']
    cfg_sha = hashlib.sha256(json.dumps({k: cfg[k] for k in PARAM_KEYS if k in cfg},
                                        sort_keys=True, ensure_ascii=False)
                             .encode()).hexdigest()[:12]

    g = np.load(f'{CACHE}/daily_grid.npz')
    ret, dates, codes = g['ret'], g['dates'], g['codes']
    close, opn = g['close'], g['open']
    paused, at_hl, at_ll = g['paused'], g['at_hlimit'], g['at_llimit']
    T, N = ret.shape
    st_g = np.load(f'{CACHE}/st_grid.npz')['is_st']
    r0 = np.nan_to_num(ret, nan=0.0)
    logc = np.cumsum(np.log1p(r0), 0)
    mom20 = np.full((T, N), np.nan, np.float32)
    mom20[MOM:] = (logc[MOM:] - logc[:-MOM]).astype(np.float32)
    r2 = r0 ** 2
    c2 = np.cumsum(r2, 0)
    vol20 = np.full((T, N), np.nan, np.float32)
    vol20[MOM:] = np.sqrt((c2[MOM:] - c2[:-MOM]) / MOM).astype(np.float32)
    tradable = (paused < 0.5) & (at_hl < 0.5) & (at_ll < 0.5) & (st_g < 0.5) & ~np.isnan(ret)
    hl = np.nan_to_num(at_hl, nan=0.0)
    chl = np.cumsum(hl, 0)
    lim250 = np.zeros((T, N), np.float32)
    lim250[WLIM:] = chl[WLIM:] - chl[:-WLIM]

    t = T - 1 if asof is None else int(np.where(dates == asof)[0][0])
    log_date = str(dates[t])

    # --- 三块打分 ---
    v = np.maximum(vol20[t] * np.sqrt(MOM), 1e-4)
    npz_p = np.load(f'{CACHE}/nets_price.npz')
    npz_d = np.load(f'{CACHE}/nets_dual.npz')
    npz_t = np.load(f'{CACHE}/nets_te.npz')
    g_p, pm_p = _gap(npz_p, t, mom20[t])
    g_d, _ = _gap(npz_d, t, mom20[t])
    g_t, _ = _gap(npz_t, t, mom20[t])

    # zspread
    RB, NB = npz_p['t'], npz_p['n']
    b = np.searchsorted(RB, t, side='right') - 1
    zs = np.full(N, np.nan, np.float32)
    if b >= 0:
        t1 = int(RB[b])
        t0 = t1 - W
        nbr = NB[b]
        idx = np.where(nbr[:, 0] >= 0)[0]
        if len(idx) >= 100:
            nb = nbr[idx]
            base = logc[t0 - 1] if t0 > 0 else np.zeros(N)
            Pw = np.exp(logc[t0:t1] - base)
            nbc = np.where(nb >= 0, nb, 0)
            sj = Pw[:, nbc.ravel()].reshape(W, len(idx), K)
            s_tr = Pw[:, idx][:, :, None] - sj
            mu, sd = s_tr.mean(0), s_tr.std(0) + 1e-9
            d = (s_tr ** 2).sum(0)
            d = d - d.min(1, keepdims=True)
            wsm = np.exp(-d) * (nb >= 0)
            wsm /= np.maximum(wsm.sum(1, keepdims=True), 1e-12)
            Pt = np.exp(logc[t] - base)
            zs[idx] = -np.nansum(((Pt[idx][:, None] - Pt[nbc] - mu) / sd) * wsm, 1)

    # ROE gap —— **方向与价格锚相反**。
    # 价格锚是 邻居 - 自身 (邻居涨了自己没跟上 = 待追赶, 正);
    # 基本面锚是 自身 - 邻居 (自己基本面比邻居好 = 质量, 正)。
    # 这里曾误用价格锚的方向, 组件差分给出 gap_roe 相关 -1.000000 才发现。
    FI = np.load(f'{CACHE}/fi_grid.npz')['fi']
    ROE = FI[:, 0]
    _mu_roe, _ = _gap(npz_p, t, ROE[t])
    g_roe = -_mu_roe
    g_roe[~np.isfinite(ROE[t])] = np.nan
    # dROE: stat_date 变化时的 ROE 环比
    droe = np.full(N, np.nan, np.float32)
    if t > 60:
        sd_now, sd_old = FI[t, 4], FI[t - 60, 4]   # 第4列=stat_date序号
        chg = np.isfinite(sd_now) & np.isfinite(sd_old) & (sd_now > sd_old)
        droe[chg] = ROE[t][chg] - ROE[t - 60][chg]

    stk = np.stack([_rank_xs(g_p / v), _rank_xs(zs), _rank_xs(g_d / v),
                    _rank_xs(g_t / v), _rank_xs(g_roe)])
    with np.errstate(all='ignore'):
        base_s = np.where(np.all(np.isnan(stk), 0), np.nan, np.nanmean(stk, 0))
    dom = (lim250[t] >= 2) & tradable[t] & ~np.isnan(g_p)
    base_s = np.where(dom, base_s, np.nan)
    hard = (paused[t] < 0.5) & (at_hl[t] < 0.5) & dom & ~np.isnan(base_s)

    TB = np.load(f'{CACHE}/tb_grid.npz')['tb']
    tbc = np.nancumsum(np.nan_to_num(TB, nan=0.0), axis=0)
    s5 = tbc[t] - tbc[t - 5]
    tm, sa, ss, so = np.maximum(s5[0], 1.0), s5[1], s5[2], s5[3]
    sweep = ss / tm
    osize = np.where(so > 0, sa / np.maximum(so, 1), np.nan)
    has = (tbc[t, 0] - tbc[t - 5, 0]) > 0
    sweep = np.where(has, sweep, np.nan)
    osize = np.where(has, osize, np.nan)
    behav = np.nanmean(np.stack([_rank01(-sweep, hard), _rank01(-osize, hard)]), 0)

    px_ok = (np.nan_to_num(close[t], nan=0) >= 2.0).astype(np.float32)
    fin = hard & np.isfinite(droe)
    droe_ok = ((np.nan_to_num(droe, nan=0) >= np.nanquantile(droe[fin], 1 / 3)).astype(np.float32)
               if fin.sum() > 200 else np.ones(N, np.float32))
    struct = (px_ok + droe_ok) / 2.0

    with np.errstate(all='ignore'):
        comb = WB * _rank01(base_s, hard) + WH * np.nan_to_num(behav, nan=0.5) + WS * struct
    comb = np.where(hard, comb, np.nan)

    # --- 选股: 缓冲带 + 上期持仓 ---
    prev_f = f'{OUTPUT_DIR}/live/last_holdings.json'
    prev = json.load(open(prev_f)) if os.path.exists(prev_f) else {}
    c2i = {c: i for i, c in enumerate(codes)}
    order = np.argsort(-np.nan_to_num(comb, nan=-1e9))
    rank = np.full(N, 1 << 30)
    rank[order] = np.arange(N)
    can_sell = (paused[t] < 0.5) & (at_ll[t] < 0.5)
    held = {c2i[c]: w for c, w in prev.items() if c in c2i}
    keep = {}
    sold = []
    for i2, wprev in held.items():
        if ((rank[i2] >= BUFFER_MULT * N_HOLD) or np.isnan(comb[i2])) and can_sell[i2]:
            sold.append(i2)
        else:
            keep[i2] = wprev
    bought = []
    for i2 in order:
        if len(keep) >= N_HOLD:
            break
        if i2 in keep or np.isnan(comb[i2]) or not hard[i2]:
            continue
        keep[i2] = 0.0
        bought.append(i2)

    ids = np.array(list(keep))
    w = np.array([np.nan_to_num(comb[i], nan=0.5) + 0.3 for i in ids])
    w = w / w.sum()

    # --- B2 约束: 个股上限 + 行业偏离带 ---
    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    months = sorted(im.keys())
    dstr = str(dates[t - 1])
    mk = [m for m in months if m <= dstr]
    imap = im[mk[-1]] if mk else {}
    inds_all = np.array([imap.get(c, '') for c in codes])
    inds = inds_all[ids]
    di = inds_all[dom]
    bench = {u: float((di == u).mean()) for u in sorted(set(di) - {''})}
    for _ in range(30):
        over = w > CAP
        if over.any():
            exc = (w[over] - CAP).sum()
            w[over] = CAP
            fr = ~over
            if fr.any() and w[fr].sum() > 0:
                w[fr] += exc * w[fr] / w[fr].sum()
        for u, bw in bench.items():
            m = inds == u
            if not m.any():
                continue
            cur = w[m].sum()
            if cur > bw + IND_BAND and cur > 0:
                cut = cur - (bw + IND_BAND)
                w[m] *= (cur - cut) / cur
                o = ~m
                if o.any() and w[o].sum() > 0:
                    w[o] += cut * w[o] / w[o].sum()
        w = np.maximum(w, 0)
        w = w / w.sum()

    # --- 输出 ---
    df = pd.DataFrame({
        'code': codes[ids], 'weight': w,
        'score': [float(comb[i]) for i in ids],
        'anchor': [float(np.nan_to_num(_rank01(base_s, hard)[i], nan=np.nan)) for i in ids],
        'behavior': [float(np.nan_to_num(behav[i], nan=0.5)) for i in ids],
        'structure': [float(struct[i]) for i in ids],
        'action': ['NEW' if i in set(bought) else 'HOLD' for i in ids],
    }).sort_values('weight', ascending=False)

    od = pd.DataFrame(
        [{'code': codes[i], 'side': 'SELL', 'weight': float(held.get(i, 0))} for i in sold] +
        [{'code': c, 'side': 'BUY', 'weight': float(x)}
         for c, x, a in zip(df['code'], df['weight'], df['action']) if a == 'NEW'])

    turn = sum(abs(dict(zip(codes[ids], w)).get(c, 0) - prev.get(c, 0))
               for c in set(list(codes[ids]) + list(prev)))
    stats = {'date': log_date, 'n_hold': int(len(ids)), 'n_buy': len(bought), 'n_sell': len(sold),
             'turnover_oneside': round(turn / 2, 4), 'max_weight': round(float(w.max()), 4),
             'domain_size': int(dom.sum()), 'candidates': int(hard.sum()),
             'config_sha': cfg_sha, 'config_version': cfg.get('version', '')}
    print(json.dumps(stats, ensure_ascii=False, indent=1))

    if write:
        od_dir = f'{OUTPUT_DIR}/live'
        os.makedirs(od_dir, exist_ok=True)
        df.to_csv(f'{od_dir}/portfolio_{log_date}.csv', index=False)
        od.to_csv(f'{od_dir}/orders_{log_date}.csv', index=False)
        json.dump(dict(zip(df['code'], df['weight'])), open(f'{od_dir}/last_holdings.json', 'w'))
        with open(f'{od_dir}/log.jsonl', 'a') as fh:
            fh.write(json.dumps(stats, ensure_ascii=False) + '\n')
        print(f'saved output/live/portfolio_{log_date}.csv (+orders, +log)')
    if return_scores:
        extra = {'comb': comb, 'hard': hard, 'codes': codes,
                 'anchor': _rank01(base_s, hard), 'behav': behav, 'struct': struct,
                 'dbg': {'v': v, 'mom20': mom20[t], 'g_p': g_p, 'zs': zs, 'g_d': g_d,
                         'g_t': g_t, 'g_roe': g_roe, 'droe': droe, 'dom': dom,
                         'lim250': lim250[t], 'tradable': tradable[t],
                         'base_s': base_s, 'sweep': sweep, 'osize': osize,
                         'behav': behav, 'struct': struct}}
        return df, od, stats, extra
    return df, od, stats


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--asof', default=None, help='YYYY-MM-DD, 默认最新交易日')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    generate(asof=a.asof, write=not a.dry_run)
