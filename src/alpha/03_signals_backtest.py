# -*- coding: utf-8 -*-
"""下游信号 + 截面回测.
信号族:
  A. zspread : 联动配对价差 z-score 聚合 (经典配对 softmax 权重, 论文公开预印本主信号, 均值回复)
  B. peer_mom: 邻居动量外溢 (经济关联动量 系, 正相关邻居的过去20日收益加权)
  C. peer_gap: peer_mom - own_mom (收敛价差)
网络: resid / raw / industry / random
评估: 周频(5日) Spearman IC + 十分组 + 多空/多头超额, 费率单边12.5bp
分段: dev 2015-01-01~2022-12-31, val 2023-01-01~2024-08-16, test>=2024-08-19 锁定不输出
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, pickle, numpy as np

CACHE = PROJ+'/cache'
OUT = PROJ+'/output/alpha'
os.makedirs(OUT, exist_ok=True)
W, REBUILD, K = 120, 21, 10
MOM = 20            # 动量回看
HOLD = 5            # 持有/调仓周期
COST = 0.00125      # 单边
DEV = ('2015-01-01','2022-12-31')
VAL = ('2023-01-01','2024-08-16')
TEST_START = '2024-08-19'

g = np.load(f'{CACHE}/daily_grid.npz')
ret, dates, codes = g['ret'], g['dates'], g['codes']
paused, at_hl, at_ll = g['paused'], g['at_hlimit'], g['at_llimit']
T, N = ret.shape
st = np.load(f'{CACHE}/st_grid.npz')['is_st']
nw = np.load(f'{CACHE}/networks.npz')
rebuilds = nw['rebuilds']
B = len(rebuilds)

ret0 = np.nan_to_num(ret, nan=0.0)
logc = np.cumsum(np.log1p(ret0), axis=0)   # 累计对数收益

# ---------- 网络字典 ----------
rng = np.random.default_rng(7)
def build_random(b):
    valid = nw['valid'][b]; idx = np.where(valid)[0]
    nbr = np.full((N,K), -1, np.int32); wgt = np.zeros((N,K), np.float32)
    for i in idx:
        pick = rng.choice(idx[idx!=i], K, replace=False)
        nbr[i] = pick; wgt[i] = 0.3
    return nbr, wgt
def build_industry(b):
    valid = nw['valid'][b]; ind = nw['ind_id'][b]
    nbr = np.full((N,K), -1, np.int32); wgt = np.zeros((N,K), np.float32)
    for u in np.unique(ind[ind>=0]):
        mem = np.where((ind==u) & valid)[0]
        if len(mem) < 2: continue
        for i in mem:
            others = mem[mem!=i]
            pick = rng.choice(others, min(K,len(others)), replace=False)
            nbr[i,:len(pick)] = pick; wgt[i,:len(pick)] = 0.3
    return nbr, wgt

NETS = {}
NETS['resid'] = (nw['nbr_resid'], nw['wgt_resid'])
NETS['raw'] = (nw['nbr_raw'], nw['wgt_raw'])
print('building industry/random nets...', flush=True)
nbr_i = np.stack([build_industry(b)[0] for b in range(B)]); 
NETS['industry'] = (nbr_i, np.where(nbr_i>=0, 0.3, 0).astype(np.float32))
nbr_rd = np.stack([build_random(b)[0] for b in range(B)])
NETS['random'] = (nbr_rd, np.where(nbr_rd>=0, 0.3, 0).astype(np.float32))

# ---------- 信号日 ----------
sig_days = np.arange(W+REBUILD, T-1, HOLD)   # 需要至少一个已完成的rebuild
def last_rebuild(t):
    return np.searchsorted(rebuilds, t, side='right') - 1

mom = np.full((T,N), np.nan, np.float32)
mom[MOM:] = (logc[MOM:] - logc[:-MOM]).astype(np.float32)
mom[:, :] = np.where(np.isnan(ret).cumsum(0) > 0*1e9, mom, mom)  # noop

def peer_signals(nbr_all, wgt_all):
    """返回 peer_mom, peer_gap (T_sig,N)"""
    S_pm = np.full((len(sig_days), N), np.nan, np.float32)
    S_pg = np.full((len(sig_days), N), np.nan, np.float32)
    for si, t in enumerate(sig_days):
        b = last_rebuild(t)
        nbr = nbr_all[b]; wgt = np.maximum(wgt_all[b], 0)
        valid = nbr[:,0] >= 0
        m = mom[t]                      # 过去MOM日动量(含t)
        nb = nbr[valid]
        wv = wgt[valid]
        pm = np.take(m, np.where(nb>=0, nb, 0))
        mask = (nb>=0) & ~np.isnan(pm)
        wv = wv * mask
        sw = wv.sum(1)
        ok = sw > 1e-9
        val = np.where(mask, np.nan_to_num(pm), 0)
        agg = (val*wv).sum(1) / np.maximum(sw, 1e-9)
        rows = np.where(valid)[0][ok]
        S_pm[si, rows] = agg[ok]
        S_pg[si, rows] = agg[ok] - m[rows]
    return S_pm, S_pg

def zspread_signal(nbr_all, wgt_all):
    S = np.full((len(sig_days), N), np.nan, np.float32)
    for b in range(B):
        t1 = rebuilds[b]; t0 = t1 - W
        t2 = rebuilds[b+1] if b+1 < B else min(t1+REBUILD, T)
        sds = [ (si,t) for si,t in enumerate(sig_days) if t1 <= t < t2 ]
        if not sds: continue
        nbr = nbr_all[b]
        valid = nbr[:,0] >= 0
        idx = np.where(valid)[0]
        nb = nbr[idx]                                # (n,K)
        # 归一化价格路径 (rebase at t0): exp(logc - logc[t0])
        seg = np.exp(logc[t0:t2] - logc[t0-1] if t0>0 else logc[t0:t2])  # (span,N)
        Pw = seg[:W]                                  # 训练窗
        nbc = np.where(nb>=0, nb, 0)
        si_p = Pw[:, idx]                             # (W,n)
        sj_p = Pw[:, nbc.ravel()].reshape(W, len(idx), -1) # (W,n,K)
        s_tr = si_p[:,:,None] - sj_p                  # (W,n,K)
        mu = s_tr.mean(0); sd = s_tr.std(0) + 1e-9
        d = (s_tr**2).sum(0)
        d = d - d.min(1, keepdims=True)
        wsm = np.exp(-d); 
        edge_ok = (nb>=0)
        wsm = wsm*edge_ok
        wsm = wsm / np.maximum(wsm.sum(1, keepdims=True), 1e-12)
        for si, t in sds:
            Pt = seg[t - t0]
            s_t = Pt[idx][:,None] - Pt[nbc]
            z = (s_t - mu)/sd
            S[si, idx] = -np.nansum(z*wsm, 1)
    return S

# ---------- 回测 ----------
fwd = np.full((T,N), np.nan, np.float32)
for t in range(T-HOLD):
    fwd[t] = (np.exp(logc[t+HOLD]-logc[t]) - 1).astype(np.float32)
# 不可交易过滤(信号日): 停牌/触板/ST/无数据
tradable = (paused < 0.5) & (at_hl < 0.5) & (at_ll < 0.5) & (st < 0.5) & ~np.isnan(ret)

import datetime
dnum = dates.astype('datetime64[D]')
def in_seg(t, seg):
    return (dnum[t] >= np.datetime64(seg[0])) and (dnum[t] <= np.datetime64(seg[1]))

def spearman_ic(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 50: return np.nan
    xr = np.argsort(np.argsort(x[m])); yr = np.argsort(np.argsort(y[m]))
    xr = (xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
    return float((xr*yr).mean())

def evaluate(S, name):
    res = {}
    for segname, seg in [('dev',DEV), ('val',VAL)]:
        ics, ls_rets, top_ex, bot_ex, uni_rets = [], [], [], [], []
        prev_top = prev_bot = None
        turns = []
        for si, t in enumerate(sig_days):
            if not in_seg(t, seg): continue
            s = S[si].copy()
            s[~tradable[t]] = np.nan
            f = fwd[t]
            ic = spearman_ic(s, f)
            if np.isnan(ic): continue
            ics.append(ic)
            m = ~np.isnan(s) & ~np.isnan(f)
            sv, fv = s[m], f[m]
            ids = np.where(m)[0]
            q = np.argsort(np.argsort(sv)) / len(sv)
            top = set(ids[q >= 0.9]); bot = set(ids[q < 0.1])
            rt = fv[q>=0.9].mean(); rb = fv[q<0.1].mean(); ru = fv.mean()
            ls_rets.append(rt-rb); top_ex.append(rt-ru); bot_ex.append(rb-ru); uni_rets.append(ru)
            if prev_top is not None and len(top):
                turns.append(1 - len(top & prev_top)/max(len(top),1))
            prev_top, prev_bot = top, bot
        ics = np.array(ics); ls = np.array(ls_rets); te = np.array(top_ex)
        ann = 252/HOLD
        to = np.mean(turns) if turns else np.nan
        # 单边换手成本: L/S 双腿, 每期换手率*2腿*2边... 简化: 每期成本= (turnover*2)*COST*2腿
        ls_net = ls - (to*2*COST*2 if to==to else 0)
        te_net = te - (to*2*COST if to==to else 0)
        def trade_stats(r):
            r = r[~np.isnan(r)]
            if len(r)==0: return {}
            win = (r>0).mean(); aw = r[r>0].mean() if (r>0).any() else 0; al = -r[r<0].mean() if (r<0).any() else 1e-9
            pf = r[r>0].sum()/max(-r[r<0].sum(),1e-9)
            return {'n':int(len(r)), 'win':round(float(win),3), 'avg_bp':round(float(r.mean()*1e4),1),
                    'wl_ratio':round(float(aw/max(al,1e-9)),2), 'profit_factor':round(float(pf),2)}
        res[segname] = {
            'IC': round(float(np.nanmean(ics)),4), 'ICIR': round(float(np.nanmean(ics)/ (np.nanstd(ics)+1e-12)),3),
            'IC_t': round(float(np.nanmean(ics)/(np.nanstd(ics)+1e-12)*np.sqrt(len(ics))),2),
            'nIC': int(len(ics)),
            'LS_ann_gross': round(float(np.nanmean(ls)*ann),4), 'LS_ann_net': round(float(np.nanmean(ls_net)*ann),4),
            'LS_sharpe_net': round(float(np.nanmean(ls_net)/(np.nanstd(ls_net)+1e-12)*np.sqrt(ann)),2),
            'TopEx_ann_net': round(float(np.nanmean(te_net)*ann),4),
            'turnover': round(float(to),3) if to==to else None,
            'LS_trades_net': trade_stats(ls_net),
        }
    print(f'== {name} ==')
    print(json.dumps(res, ensure_ascii=False))
    return res

all_res = {}
for netname, (nbr_all, wgt_all) in NETS.items():
    print(f'--- network: {netname} ---', flush=True)
    pm, pg = peer_signals(nbr_all, wgt_all)
    zs = zspread_signal(nbr_all, wgt_all)
    all_res[f'{netname}:peer_mom'] = evaluate(pm, f'{netname}:peer_mom')
    all_res[f'{netname}:peer_gap'] = evaluate(pg, f'{netname}:peer_gap')
    all_res[f'{netname}:zspread'] = evaluate(zs, f'{netname}:zspread')
    np.savez_compressed(f'{OUT}/signals_{netname}.npz', sig_days=sig_days, peer_mom=pm, peer_gap=pg, zspread=zs)

with open(f'{OUT}/metrics_v1.json','w') as fh:
    json.dump(all_res, fh, ensure_ascii=False, indent=1)
print('done')
