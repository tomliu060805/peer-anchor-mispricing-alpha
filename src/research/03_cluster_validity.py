# -*- coding: utf-8 -*-
"""聚类有效性: 两个避免循环论证的检验。

02 号脚本复现了"聚类组内离散度低于行业分组", 但那个判据有循环性——
K-means 的目标函数就是最小化这些比率上的组内方差, 用组内离散度给它打分,
等于用它自己的目标函数当判据。随机分组基准 0.996 接近 1, 赢它没有难度。

两个不循环的判据:

  A. 留一比率  用 16 个比率聚类, 在**没参与聚类**的第 17 个比率上比组内离散度。
     若分组只是拟合了噪声, 在未见过的维度上不该比行业分组更紧。

  B. 跨期持续  在 t 时刻定下分组, 一年后用**同一批分组**(成员不变)重新测离散度。
     若抓到的是持续的财务结构而非当期噪声, 一年后应仍紧于行业分组。

两个判据都对随机分组零基准报告。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEED, K = 20260902, 9
rng = np.random.default_rng(SEED)
_Z = None


def _z():
    global _Z
    if _Z is None:
        _Z = np.load(f'{CACHE}/finratio_grid.npz')
    return _Z


def cross(ds):
    z = _z()
    R, names, dates, codes = z['ratios'], list(z['names']), z['dates'], z['codes']
    t = int(np.where(dates == ds)[0][0])
    X = R[t].T
    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    mk = [m for m in sorted(im) if m <= str(dates[t - 1])]
    imap = im[mk[-1]] if mk else {}
    y = np.array([imap.get(c, '') for c in codes])
    ok = (y != '') & (np.isfinite(X).sum(1) >= len(names) - 3)
    return X[ok], y[ok], codes[ok], [str(n) for n in names]


def prep(X, cols=None):
    X = X[:, cols] if cols is not None else X.copy()
    X = X.astype(np.float64)
    med = np.nanmedian(X, 0)
    ix = np.where(~np.isfinite(X))
    X[ix] = np.take(med, ix[1])
    r = np.argsort(np.argsort(X, 0), 0) / max(len(X) - 1, 1)
    return StandardScaler().fit_transform(r)


def disp_one(v, lab):
    """单个比率上的组内 MAD(规模加权) / 全样本 MAD。

    先做截面秩变换再测量, 与 02 号脚本口径一致。原始比率的分布极偏
    (利息保障、周转率的尾部能差几个数量级), 直接用原始值时 MAD 由尺度主导,
    结论会随口径翻转——recv_turn / inv_turn 就是这样在两个脚本间给出相反判定的。
    """
    m = np.isfinite(v)
    v, lab = v[m], lab[m]
    if len(v) < 50:
        return np.nan
    v = np.argsort(np.argsort(v)).astype(np.float64) / max(len(v) - 1, 1)
    tot = np.median(np.abs(v - np.median(v))) + 1e-12
    num = den = 0.0
    for u in np.unique(lab):
        s = lab == u
        if s.sum() < 5:
            continue
        num += s.sum() * np.median(np.abs(v[s] - np.median(v[s])))
        den += s.sum()
    return num / max(den, 1) / tot


def rand_like(lab):
    sizes = np.bincount(lab)
    rl = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    rng.shuffle(rl)
    return rl


def test_loro(ds):
    """A. 留一比率。"""
    X0, y, codes, names = cross(ds)
    print(f'\n=== A. 留一比率 ({ds}, {len(y)} 只) ===')
    print('  用其余16个比率聚类, 在被留出的那个比率上比离散度\n')
    print(f'{"留出的比率":14s} {"聚类":>8s} {"行业":>8s} {"随机":>8s}   判定')
    win_i = win_r = n = 0
    rows = []
    for j, nm in enumerate(names):
        cols = [k for k in range(len(names)) if k != j]
        lab = KMeans(K, n_init=10, random_state=SEED).fit(prep(X0, cols)).labels_
        v = X0[:, j].astype(np.float64)
        dc, di = disp_one(v, lab), disp_one(v, y)
        dr = float(np.mean([disp_one(v, rand_like(lab)) for _ in range(20)]))
        if not np.isfinite(dc):
            continue
        n += 1
        wi, wr = dc < di, dc < dr
        win_i += wi
        win_r += wr
        rows.append({'ratio': nm, 'cluster': dc, 'industry': di, 'random': dr})
        print(f'{nm:14s} {dc:>8.3f} {di:>8.3f} {dr:>8.3f}   '
              f'{"聚类更紧" if wi else "行业更紧"}{"" if wr else " (不如随机)"}')
    print(f'\n  留出维度上 聚类紧于行业: {win_i}/{n}')
    print(f'  留出维度上 聚类紧于随机: {win_r}/{n}   <- 这条不成立则分组只是拟合噪声')
    return {'win_vs_industry': int(win_i), 'win_vs_random': int(win_r), 'n': n, 'rows': rows}


def test_persist(ds0, ds1):
    """B. 跨期持续: ds0 定分组, ds1 测离散度(成员不变)。"""
    X0, y0, c0, names = cross(ds0)
    X1, y1, c1, _ = cross(ds1)
    lab0 = KMeans(K, n_init=10, random_state=SEED).fit(prep(X0)).labels_
    m = {c: l for c, l in zip(c0, lab0)}
    keep = np.array([c in m for c in c1])
    X1, y1, c1 = X1[keep], y1[keep], c1[keep]
    lab1 = np.array([m[c] for c in c1])
    print(f'\n=== B. 跨期持续 ({ds0} 定分组 -> {ds1} 测量, {len(c1)} 只仍在样本内) ===')
    print(f'{"比率":14s} {"聚类":>8s} {"行业":>8s} {"随机":>8s}   判定')
    win_i = win_r = n = 0
    rows = []
    for j, nm in enumerate(names):
        v = X1[:, j].astype(np.float64)
        dc, di = disp_one(v, lab1), disp_one(v, y1)
        dr = float(np.mean([disp_one(v, rand_like(lab1)) for _ in range(20)]))
        if not np.isfinite(dc):
            continue
        n += 1
        wi, wr = dc < di, dc < dr
        win_i += wi
        win_r += wr
        rows.append({'ratio': nm, 'cluster': dc, 'industry': di, 'random': dr})
        print(f'{nm:14s} {dc:>8.3f} {di:>8.3f} {dr:>8.3f}   '
              f'{"聚类更紧" if wi else "行业更紧"}{"" if wr else " (不如随机)"}')
    print(f'\n  一年后 聚类紧于行业: {win_i}/{n}')
    print(f'  一年后 聚类紧于随机: {win_r}/{n}   <- 这条说明结构是否持续')
    return {'win_vs_industry': int(win_i), 'win_vs_random': int(win_r), 'n': n, 'rows': rows}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='2021-12-31')
    ap.add_argument('--later', default='2022-12-30')
    a = ap.parse_args()
    out = {'fit_date': a.date, 'later_date': a.later, 'k': K,
           'leave_one_ratio_out': test_loro(a.date),
           'persistence': test_persist(a.date, a.later)}
    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump(out, open(f'{OUTPUT_DIR}/research/cluster_validity.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\n已写 output/research/cluster_validity.json')
