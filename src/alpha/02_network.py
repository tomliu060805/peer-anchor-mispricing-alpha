# -*- coding: utf-8 -*-
"""联动网络构建: 每21个交易日重建一次.
三种网络: resid(剔除市场+行业后残差相关) / raw(原始相关) / industry(同行业) / random.
存 topK 邻居索引+权重, 供下游信号使用."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, pickle, numpy as np

CACHE = PROJ+'/cache'
W = 120          # 相关窗口
REBUILD = 21     # 重建频率(交易日)
K = 10           # top-K 邻居
MINVALID = 110   # 窗口内最少有效天数

def main():
    g = np.load(f'{CACHE}/daily_grid.npz')
    ret = g['ret']; dates = g['dates']; codes = g['codes']
    T, N = ret.shape
    st = np.load(f'{CACHE}/st_grid.npz')['is_st']
    with open(f'{CACHE}/industry_monthly.pkl','rb') as fh:
        ind_map = pickle.load(fh)
    ind_months = sorted(ind_map.keys())

    rebuilds = list(range(W, T, REBUILD))
    B = len(rebuilds)
    print(f'T={T} N={N} rebuilds={B}')
    nbr_resid = np.full((B, N, K), -1, np.int32)
    wgt_resid = np.zeros((B, N, K), np.float32)
    nbr_raw = np.full((B, N, K), -1, np.int32)
    wgt_raw = np.zeros((B, N, K), np.float32)
    ind_id_all = np.full((B, N), -1, np.int16)
    valid_all = np.zeros((B, N), bool)

    rng = np.random.default_rng(42)
    for b, t1 in enumerate(rebuilds):
        Rw = ret[t1-W:t1]                       # (W,N) 截止 t1-1 收盘, t1 为重建日(用 t1-1 前数据? )
        valid = (~np.isnan(Rw)).sum(0) >= MINVALID
        # ST 或近端全停牌的剔除
        st_now = st[t1-1]; valid &= ~(st_now == 1)
        # 行业 (PIT: 用重建日之前最近的月度快照)
        dstr = str(dates[t1-1])
        m = [mm for mm in ind_months if mm <= dstr]
        imap = ind_map[m[-1]] if m else {}
        inds = np.array([imap.get(c, '') for c in codes])
        uniq = sorted(set(inds[valid]) - {''})
        ind_id = np.full(N, -1, np.int16)
        for i, u in enumerate(uniq):
            ind_id[inds == u] = i
        ind_id_all[b] = ind_id
        valid &= (ind_id >= 0)
        idx = np.where(valid)[0]
        n = len(idx); valid_all[b] = valid
        X = np.nan_to_num(Rw[:, idx], nan=0.0).astype(np.float64)

        # 残差: 剔除市场(等权均值)与行业均值
        mkt = X.mean(1, keepdims=True)
        beta = (X * mkt).sum(0) / np.maximum((mkt * mkt).sum(), 1e-12)
        Xr = X - mkt @ beta[None, :]
        gid = ind_id[idx]
        for u in np.unique(gid):
            sel = gid == u
            Xr[:, sel] -= Xr[:, sel].mean(1, keepdims=True)

        def topk_corr(M):
            sd = M.std(0) + 1e-12
            Z = (M - M.mean(0)) / sd
            C = (Z.T @ Z / len(M)).astype(np.float32)
            np.fill_diagonal(C, -9)
            kk = min(K, n-1)
            part = np.argpartition(-C, kk, axis=1)[:, :kk]
            rows = np.arange(n)[:, None]
            vals = C[rows, part]
            order = np.argsort(-vals, axis=1)
            return part[rows, order], vals[rows, order]

        nb_r, wv_r = topk_corr(Xr)
        nb_o, wv_o = topk_corr(X - mkt)   # raw: 只剔市场
        kk = nb_r.shape[1]
        nbr_resid[b, idx, :kk] = idx[nb_r]; wgt_resid[b, idx, :kk] = wv_r
        nbr_raw[b, idx, :kk] = idx[nb_o]; wgt_raw[b, idx, :kk] = wv_o
        if b % 20 == 0:
            print(f'  b={b} date={dstr} n={n} medcorr_resid={np.median(wv_r):.3f} medcorr_raw={np.median(wv_o):.3f}', flush=True)

    np.savez_compressed(f'{CACHE}/networks.npz',
        rebuilds=np.array(rebuilds, np.int32), dates=dates, codes=codes,
        nbr_resid=nbr_resid, wgt_resid=wgt_resid,
        nbr_raw=nbr_raw, wgt_raw=wgt_raw,
        ind_id=ind_id_all, valid=valid_all)
    print('saved networks.npz')

if __name__ == '__main__':
    main()
