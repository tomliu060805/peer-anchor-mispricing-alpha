# -*- coding: utf-8 -*-
"""财务结构网络: 按财务比率相似度连边, 与价格/价量/转移熵三张网络同构。

连边口径
--------
在每个重建点 t1, 取 t1-1 日可得的 17 个比率(已 PIT), 做截面秩变换并标准化,
两股之间的距离 = 该 17 维空间的欧氏距离, 取最近 K=5 只为邻居,
权重 w = exp(-(d/d_中位)^2) —— 平滑核, 落在 (0,1], 与其余三张网络用相关系数
作权重时的取值范围可比。

**不扣行业均值。** 三张收益网络扣掉行业均值是为了让残差"行业之外";
这里刻意保留, 因为要检验的正是"财务结构相似度与行业有多大重合"——
先验地扣掉就把待检验的东西假设掉了。重合度作为诊断输出。

重建点与域的判定与 pipeline.stage_networks 完全一致(同一批 valid 股票),
否则后续与其余锚做正交性比较时不同域会污染结论。

产出 cache/nets_fin.npz: t / n / w, 结构与 nets_price.npz 相同。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'live'))
from paths import CACHE_DIR as CACHE

import os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

W, K, WLIM, REBUILD = 120, 5, 250, 21
N_WORKERS = int(os.environ.get('N_WORKERS', '80'))
_C = {}


def _ctx():
    if not _C:
        g = np.load(f'{CACHE}/daily_grid.npz')
        _C['ret'] = g['ret']
        _C['st'] = np.load(f'{CACHE}/st_grid.npz')['is_st']
        z = np.load(f'{CACHE}/finratio_grid.npz')
        _C['fr'] = z['ratios']
        _C['nr'] = len(z['names'])
    return _C


def _one(t1):
    c = _ctx()
    ret, st_g, FR, NR = c['ret'], c['st'], c['fr'], c['nr']
    Nn = ret.shape[1]
    on = np.full((Nn, K), -1, np.int32)
    ow = np.zeros((Nn, K), np.float32)

    # 与 stage_networks 相同的域判定
    Rw = ret[t1 - W:t1]
    valid = (~np.isnan(Rw)).sum(0) >= 110
    valid &= ~(st_g[t1 - 1] == 1)
    X = FR[t1 - 1].T.astype(np.float64)                  # (N, R), PIT: 用 t1-1
    valid &= np.isfinite(X).sum(1) >= NR - 3
    idx = np.where(valid)[0]
    n = len(idx)
    if n < 200:
        return t1, on, ow

    Xi = X[idx]
    med = np.nanmedian(Xi, 0)
    bad = np.where(~np.isfinite(Xi))
    Xi[bad] = np.take(med, bad[1])
    R = np.argsort(np.argsort(Xi, 0), 0) / max(n - 1, 1)  # 截面秩 -> [0,1]
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)

    # 欧氏距离平方: |a|^2 + |b|^2 - 2ab
    sq = (R * R).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * (R @ R.T)
    np.fill_diagonal(D, np.inf)
    np.maximum(D, 0, out=D)

    part = np.argpartition(D, K, axis=1)[:, :K]
    dv = np.take_along_axis(D, part, 1)
    o = np.argsort(dv, 1)
    nb = np.take_along_axis(part, o, 1)
    dd = np.sqrt(np.take_along_axis(dv, o, 1))
    scale = np.median(dd) + 1e-12
    wv = np.exp(-(dd / scale) ** 2).astype(np.float32)

    on[idx] = idx[nb].astype(np.int32)
    ow[idx] = wv
    return t1, on, ow


def main():
    c = _ctx()
    T = c['ret'].shape[0]
    rb = list(range(WLIM, T, REBUILD))
    print(f'财务结构网络: {len(rb)} 个重建点, K={K}', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        outs = sorted(ex.map(_one, rb, chunksize=1))
    np.savez_compressed(f'{CACHE}/nets_fin.npz',
                        t=np.array([o[0] for o in outs], np.int32),
                        n=np.stack([o[1] for o in outs]),
                        w=np.stack([o[2] for o in outs]))
    nz = sum(int((o[1][:, 0] >= 0).sum()) for o in outs)
    print(f'完成 ({time.time()-t0:.0f}s), 平均每期有邻居的股票 {nz/len(outs):.0f} 只')


if __name__ == '__main__':
    main()
