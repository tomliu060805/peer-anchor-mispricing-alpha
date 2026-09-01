# -*- coding: utf-8 -*-
"""复现: 行业分类能否解释财务结构 (A股口径)。

对应原工作的两个结论:
  (1) 监督学习从财务比率反推行业标签, 验证准确率 49.3%
  (2) 无监督聚类得到 9 个跨行业的财务结构分组, 多数比率下组内离散度低于行业分组

两处必须自己补的判据 —— 原文没给, 但没有它们数字无法解读:
  * 准确率要对**基线**。标普500分11个板块, 随机基线约 9%; 49.3% 是随机的 5.5 倍,
    说"行业只解释一部分"公允, 说"49.3% 很低"则缺基线。本脚本同时报告
    随机基线与"最大类占比"基线(always-predict-majority)。
  * 组内离散度要对**随机分组**。把 N 只股票随机切成同样大小的 K 组, 组内离散度
    自然低于全样本; 不比随机分组就无法判断聚类是否真的抓到结构。

分段: 只用 dev 段(2016-2022)拟合与选参, val 段仅报告, test 段不碰。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 20260902
K_CLUSTER = 9                     # 与原工作一致
rng = np.random.default_rng(SEED)


def load(ds):
    z = np.load(f'{CACHE}/finratio_grid.npz')
    R, names, dates, codes = z['ratios'], list(z['names']), z['dates'], z['codes']
    t = int(np.where(dates == ds)[0][0])
    X = R[t].T                                        # (N, R)
    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    mk = [m for m in sorted(im) if m <= str(dates[t - 1])]
    imap = im[mk[-1]] if mk else {}
    y = np.array([imap.get(c, '') for c in codes])
    # 只保留行业已知且比率覆盖足够的股票
    ok = (y != '') & (np.isfinite(X).sum(1) >= len(names) - 3)
    return X[ok], y[ok], codes[ok], names


def prep(X):
    """列中位数填缺 -> 分位数变换(抗极值) -> 标准化。"""
    X = X.copy()
    med = np.nanmedian(X, 0)
    ix = np.where(~np.isfinite(X))
    X[ix] = np.take(med, ix[1])
    r = np.argsort(np.argsort(X, 0), 0) / max(len(X) - 1, 1)
    return StandardScaler().fit_transform(r)


def supervised(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    out = {}
    for nm, m in [('随机森林', RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                  random_state=SEED, n_jobs=8)),
                  ('多项logistic', LogisticRegression(max_iter=2000, C=1.0,
                                                     multi_class='multinomial'))]:
        m.fit(Xtr, ytr)
        out[nm] = float((m.predict(Xte) == yte).mean())
    u, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    out['基线_随机猜'] = float((p ** 2).sum())          # 按类先验随机猜的期望准确率
    out['基线_全猜最大类'] = float(p.max())
    out['行业数'] = int(len(u))
    out['样本数'] = int(len(y))
    return out


def dispersion(X, lab):
    """各比率的组内离散度: 组内 MAD 的规模加权均值 / 全样本 MAD。越小越紧。"""
    res = []
    for j in range(X.shape[1]):
        v = X[:, j]
        tot = np.median(np.abs(v - np.median(v))) + 1e-12
        num = den = 0.0
        for u in np.unique(lab):
            m = lab == u
            if m.sum() < 5:
                continue
            num += m.sum() * np.median(np.abs(v[m] - np.median(v[m])))
            den += m.sum()
        res.append(num / max(den, 1) / tot)
    return np.array(res)


def main(ds='2022-12-30'):
    X0, y, codes, names = load(ds)
    X = prep(X0)
    print(f'截面 {ds}: {len(y)} 只股票, {X.shape[1]} 个比率, {len(set(y))} 个行业\n')

    sup = supervised(X, y)
    print('=== (1) 从财务比率反推行业标签 ===')
    for k in ['随机森林', '多项logistic']:
        print(f'  {k:12s} 验证准确率 {sup[k]:.3f}')
    print(f'  {"基线: 按先验随机猜":12s} {sup["基线_随机猜"]:.3f}')
    print(f'  {"基线: 全猜最大行业":12s} {sup["基线_全猜最大类"]:.3f}')
    best = max(sup['随机森林'], sup['多项logistic'])
    print(f'  -> 最优模型是随机基线的 {best/sup["基线_随机猜"]:.1f} 倍, '
          f'是最大类基线的 {best/sup["基线_全猜最大类"]:.1f} 倍\n')

    km = KMeans(K_CLUSTER, n_init=10, random_state=SEED).fit(X)
    lab = km.labels_
    d_clu = dispersion(X, lab)
    d_ind = dispersion(X, y)
    # 随机分组零基准: 组大小与聚类相同, 只是成员随机
    sizes = np.bincount(lab)
    d_rnd = []
    for _ in range(50):
        rl = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
        rng.shuffle(rl)
        d_rnd.append(dispersion(X, rl))
    d_rnd = np.mean(d_rnd, 0)

    print(f'=== (2) 组内离散度 (越小越紧, 已除以全样本 MAD) ===')
    print(f'{"比率":14s} {"聚类":>8s} {"行业":>8s} {"随机分组":>9s}   判定')
    win_ind = win_rnd = 0
    for j, n in enumerate(names):
        wi = d_clu[j] < d_ind[j]
        wr = d_clu[j] < d_rnd[j]
        win_ind += wi
        win_rnd += wr
        tag = ('聚类更紧' if wi else '行业更紧') + ('' if wr else ' (但不如随机分组)')
        print(f'{str(n):14s} {d_clu[j]:>8.3f} {d_ind[j]:>8.3f} {d_rnd[j]:>9.3f}   {tag}')
    print(f'\n  聚类紧于行业: {win_ind}/{len(names)} 个比率')
    print(f'  聚类紧于随机分组: {win_rnd}/{len(names)} 个比率  <- 这一行才说明聚类有没有抓到结构')

    # 聚类与行业的交叉度
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(y, lab)
    nmi = normalized_mutual_info_score(y, lab)
    print(f'\n  聚类 vs 行业: ARI {ari:.3f}  NMI {nmi:.3f}  (0=无关, 1=完全一致)')

    res = {'date': ds, 'supervised': sup,
           'disp_cluster': d_clu.tolist(), 'disp_industry': d_ind.tolist(),
           'disp_random': d_rnd.tolist(), 'names': [str(n) for n in names],
           'win_vs_industry': int(win_ind), 'win_vs_random': int(win_rnd),
           'ari': float(ari), 'nmi': float(nmi), 'k': K_CLUSTER}
    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump(res, open(f'{OUTPUT_DIR}/research/replicate_{ds}.json', 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n已写 output/research/replicate_{ds}.json')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='2022-12-30', help='截面日期(默认 dev 段末)')
    main(ap.parse_args().date)
