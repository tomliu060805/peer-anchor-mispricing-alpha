# -*- coding: utf-8 -*-
"""联合因子-图框架在 A 股的复现。

这条线**不接生产策略**。生产策略用的是残差分解的"特质"那一半(邻居残差共动 ->
错价追赶); 本文检验的是"系统性"那一半能做什么 —— 用因子暴露相似度组织资产的
局部关系, 服务于分散化。两者目标相反, 结论互不影响。

复现的核心对照(原工作最有解释力的那一条): **保持配置算法不变, 只替换协方差估计**
    样本协方差 / 普通因子协方差 / 联合估计协方差
再加两条原工作没有的:
    (a) **随机图零基准** —— 把图的节点标签随机置换, 度数分布完全保留, 只打乱
        "哪只股票处在网络的什么位置"。这才能回答"图位置本身是否携带信息"。
    (b) 逐段报告 + 完整单笔指标

与原设定的偏离(数据所限, 必须说明):
    * 股票池: A 股(单市场), 原工作是三个市场各 100 只; 规避幸存者偏差,
      本文按滚动流动性重选而非固定某年名单。
    * 正则强度: 原工作未给。按**结构目标**(平均度数)标定, 不按样本外表现调 ——
      后者正是原工作自己提示要核查的泄漏来源。
    * 因子数 6 / 叶子簇 24 / 半衰期一年: 沿用原设定, 不调。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(_HERE, '..'))
_sys.path.insert(0, _HERE)
from paths import CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, pickle, time
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import mingle as MG

N_UNIV = 300            # 股票池规模, 对齐原工作的 300 只
K_FACTOR = 6            # 潜在因子数(原设定)
N_LEAVES = 24           # 叶子簇数(原设定)
HALFLIFE = 252          # 一年半衰期(原设定)
TRAIN_D = 504           # 两年训练窗(原设定)
COST = 0.0020           # 双边 20bp(原工作与本项目同口径)
N_RAND = 5              # 随机图零基准抽样次数
SEG = {'2018-2020': ('2018-01-01', '2020-12-31'),
       '2021-2023': ('2021-01-01', '2023-12-31'),
       '2024-2026': ('2024-01-01', '2026-12-31')}
N_WORKERS = int(os.environ.get('N_WORKERS', '40'))
_G = {}


def _load():
    if _G:
        return _G
    g = np.load(f'{CACHE}/daily_grid.npz')
    ret, dates, codes, money = g['ret'], g['dates'], g['codes'], g['money']
    st = np.load(f'{CACHE}/st_grid.npz')['is_st']
    paused = g['paused']
    T, N = ret.shape
    ok = (paused < 0.5) & (st < 0.5) & np.isfinite(ret)
    _G.update(ret=ret, dates=dates, codes=codes, money=money, ok=ok, T=T, N=N)
    return _G


def rebalance_days():
    """每月最后一个交易日重估。"""
    d = _load()['dates']
    ym = np.array([str(x)[:7] for x in d])
    idx = []
    for i in range(1, len(ym)):
        if ym[i] != ym[i - 1]:
            idx.append(i - 1)
    idx.append(len(ym) - 1)
    return np.array(idx)


def universe_at(t):
    """按过去 250 日中位成交额取前 N_UNIV, 全程 PIT。"""
    d = _load()
    lo = max(0, t - 250)
    mn = np.nanmedian(np.where(d['ok'][lo:t + 1], d['money'][lo:t + 1], np.nan), 0)
    cov = d['ok'][lo:t + 1].mean(0)
    mn = np.where(cov > 0.9, mn, np.nan)
    order = np.argsort(-np.nan_to_num(mn, nan=-1.0))
    return np.sort(order[:N_UNIV])


def _one_period(args):
    t, t_next, seed = args
    d = _load()
    ret = d['ret']
    uni = universe_at(t)
    R = ret[t - TRAIN_D + 1:t + 1, uni].T.astype(np.float64)
    R = np.nan_to_num(R, nan=0.0)
    R = R - R.mean(1, keepdims=True)

    S_sam = MG.sample_cov(R, HALFLIFE)
    S_pca = MG.pca_factor_cov(R, K_FACTOR, HALFLIFE)
    B, F, W, Psi, info = MG.fit_mingle(R, K=K_FACTOR, halflife=HALFLIFE, seed=seed)
    S_mgl = MG.factor_cov(B, F, Psi, HALFLIFE)

    fwd = ret[t + 1:t_next + 1, uni]
    fwd = np.nan_to_num(fwd, nan=0.0)
    per = np.prod(1.0 + fwd, 0) - 1.0                      # 持有期简单收益

    out = {'t': int(t), 'uni': uni, 'per': per,
           'cond_sam': float(np.linalg.cond(S_sam)),
           'cond_mgl': float(np.linalg.cond(S_mgl)),
           'info': info}

    rng = np.random.default_rng(seed + 991)
    for nm, S in (('sam', S_sam), ('pca', S_pca), ('mgl', S_mgl)):
        leaves, frac = MG.spectral_leaves(S, N_LEAVES)
        out[f'w_{nm}'] = MG.allocate(S, leaves)
        out[f'fiedler_{nm}'] = frac
        lab = np.zeros(len(uni), np.int32)
        for ci, lf in enumerate(leaves):
            lab[lf] = ci
        out[f'lab_{nm}'] = lab
        if nm == 'mgl':
            out['w_cc'] = MG.allocate(S, leaves, MG.contagion_intra(W, leaves))
            # 随机图零基准: 置换节点标签, 度数分布完全保留
            for r in range(N_RAND):
                pm = rng.permutation(len(uni))
                Wr = W[np.ix_(pm, pm)]
                out[f'w_cc_rand{r}'] = MG.allocate(S, leaves, MG.contagion_intra(Wr, leaves))
    out['w_eq'] = np.full(len(uni), 1.0 / len(uni))
    # 全局最小方差(多头约束, 投影梯度) 作为外部对照
    out['w_mv'] = _minvar(S_sam)
    out['ind_ratio_mgl'] = _industry_ratio(W, d['codes'][uni], d['dates'][t - 1])
    out['ind_ratio_cor'] = _industry_ratio(np.abs(np.corrcoef(R)), d['codes'][uni],
                                           d['dates'][t - 1])
    return out


def _minvar(S, iters=500):
    N = S.shape[0]
    w = np.full(N, 1.0 / N)
    L = np.linalg.eigvalsh(S).max() * 2 + 1e-12
    for _ in range(iters):
        w = w - (S @ w) / L
        w = np.maximum(w, 0)
        s = w.sum()
        w = w / s if s > 1e-12 else np.full(N, 1.0 / N)
    return w


_IND = {}


def _industry_ratio(W, codes, dstr):
    """行业内平均边权 / 行业间平均边权 —— 原工作的事后检验口径。"""
    if not _IND:
        with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
            _IND['m'] = pickle.load(fh)
            _IND['k'] = sorted(_IND['m'])
    mk = [m for m in _IND['k'] if m <= str(dstr)]
    imap = _IND['m'][mk[-1]] if mk else {}
    lab = np.array([imap.get(c, '') for c in codes])
    ok = lab != ''
    if ok.sum() < 50:
        return np.nan
    Wx, lx = W[np.ix_(ok, ok)], lab[ok]
    same = lx[:, None] == lx[None, :]
    np.fill_diagonal(same, False)
    off = ~same
    np.fill_diagonal(off, False)
    a, b = Wx[same].mean(), Wx[off].mean()
    return float(a / b) if b > 1e-12 else np.nan


def main():
    d = _load()
    rb = rebalance_days()
    start = np.searchsorted(d['dates'], '2018-01-01')
    pts = [(int(rb[i]), int(rb[i + 1]), i) for i in range(len(rb) - 1)
           if rb[i] >= max(start, TRAIN_D + 10)]
    print(f'重估点 {len(pts)} 个: {d["dates"][pts[0][0]]} -> {d["dates"][pts[-1][0]]}', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        res = list(ex.map(_one_period, pts, chunksize=1))
    print(f'估计完成 ({time.time() - t0:.0f}s)', flush=True)

    keys = ['w_eq', 'w_mv', 'w_sam', 'w_pca', 'w_mgl', 'w_cc'] + \
           [f'w_cc_rand{r}' for r in range(N_RAND)]
    recs = {k: [] for k in keys}
    turns = {k: [] for k in keys}
    prev = {k: None for k in keys}
    prev_uni = None
    diag = {'cond_sam': [], 'cond_mgl': [], 'fiedler_sam': [], 'fiedler_mgl': [],
            'ind_ratio_mgl': [], 'ind_ratio_cor': [], 'avg_degree': []}
    dts = []
    for o in res:
        dts.append(str(d['dates'][o['t']]))
        for k in diag:
            diag[k].append(o['info']['avg_degree'] if k == 'avg_degree' else o.get(k, np.nan))
        for k in keys:
            w = o[k]
            if prev[k] is None or prev_uni is None:
                turn = 1.0
            else:
                m_old = dict(zip(prev_uni, prev[k]))
                turn = sum(abs(w[i] - m_old.get(c, 0.0))
                           for i, c in enumerate(o['uni'])) + \
                       sum(v for c, v in m_old.items() if c not in set(o['uni'].tolist()))
                turn = turn / 2.0
            r = float(w @ o['per']) - turn * COST
            recs[k].append(r)
            turns[k].append(turn)
            prev[k] = w
        prev_uni = o['uni']

    # 相邻期聚类稳定性(仅对两期共有的股票计算)
    from sklearn.metrics import normalized_mutual_info_score as _nmi
    nmis = {'sam': [], 'mgl': []}
    for i in range(1, len(res)):
        a, b = res[i - 1], res[i]
        common, ia, ib = np.intersect1d(a['uni'], b['uni'], return_indices=True)
        if len(common) < 50:
            continue
        for nm in nmis:
            nmis[nm].append(_nmi(a[f'lab_{nm}'][ia], b[f'lab_{nm}'][ib]))
    out_nmi = {k: float(np.mean(v)) for k, v in nmis.items() if v}

    dts = np.array(dts, dtype='datetime64[D]')
    NAMES = {'w_eq': '等权基准', 'w_mv': '最小方差(多头)', 'w_sam': '谱切分:样本协方差',
             'w_pca': '谱切分:普通因子协方差', 'w_mgl': '谱切分:联合估计协方差',
             'w_cc': '联合估计 + 簇内度数倒数'}

    def stat(v, m):
        v = np.asarray(v)[m]
        if len(v) < 12:
            return None
        nav = np.cumprod(1 + v)
        ann = float(nav[-1] ** (12 / len(v)) - 1)
        sh = float(v.mean() / (v.std(ddof=1) + 1e-12) * np.sqrt(12))
        mdd = float((1 - nav / np.maximum.accumulate(nav)).max())
        win, loss = v[v > 0], v[v < 0]
        return {'CAGR': ann, 'Sharpe': sh, 'MDD': mdd, 'n': int(len(v)),
                'winrate': float((v > 0).mean()), 'avg': float(v.mean()),
                'pl': float(win.mean() / abs(loss.mean())) if len(loss) else np.nan,
                'pf': float(win.sum() / abs(loss.sum())) if len(loss) else np.nan}

    out = {'periods': len(res), 'diag': {}, 'full': {}, 'seg': {}}
    print(f'\n{"="*100}\n全样本 ({dts[0]} ~ {dts[-1]}, {len(res)} 期月频, 双边20bp)\n{"="*100}')
    print(f'{"策略":26s} {"年化":>8s} {"夏普":>7s} {"MDD":>8s} {"期数":>5s} '
          f'{"胜率":>7s} {"单期均":>8s} {"盈亏比":>7s} {"盈利因子":>8s} {"月换手":>7s}')
    mall = np.ones(len(res), bool)
    for k, nm in NAMES.items():
        s = stat(recs[k], mall)
        s['turn'] = float(np.mean(turns[k]))
        out['full'][nm] = s
        print(f'{nm:26s} {s["CAGR"]:8.2%} {s["Sharpe"]:7.3f} {s["MDD"]:8.2%} {s["n"]:5d} '
              f'{s["winrate"]:6.1%} {s["avg"]:7.3%} {s["pl"]:7.2f} {s["pf"]:8.2f} {s["turn"]:6.1%}')

    rr = [stat(recs[f'w_cc_rand{r}'], mall) for r in range(N_RAND)]
    ca = np.array([x['CAGR'] for x in rr]); sa = np.array([x['Sharpe'] for x in rr])
    cc = out['full']['联合估计 + 簇内度数倒数']
    zc = (cc['CAGR'] - ca.mean()) / (ca.std(ddof=1) + 1e-12)
    zs = (cc['Sharpe'] - sa.mean()) / (sa.std(ddof=1) + 1e-12)
    print(f'\n★ 随机图零基准({N_RAND}次, 度数分布完全保留, 只打乱哪只股票在什么位置)')
    print(f'  年化 {ca.mean():+.2%} ± {ca.std(ddof=1):.2%}   夏普 {sa.mean():.3f} ± {sa.std(ddof=1):.3f}')
    print(f'  -> 簇内度数倒数超出 {zc:+.1f}σ (年化) / {zs:+.1f}σ (夏普)')
    out['zero_baseline'] = {'CAGR_mean': float(ca.mean()), 'CAGR_std': float(ca.std(ddof=1)),
                            'Sharpe_mean': float(sa.mean()), 'Sharpe_std': float(sa.std(ddof=1)),
                            'z_CAGR': float(zc), 'z_Sharpe': float(zs), 'n_draw': N_RAND}

    print(f'\n{"="*100}\n分段\n{"="*100}')
    print(f'{"策略":26s}' + ''.join(f'{k:>22s}' for k in SEG))
    for k, nm in NAMES.items():
        row = f'{nm:26s}'
        out['seg'][nm] = {}
        for sk, (a, b) in SEG.items():
            m = (dts >= np.datetime64(a)) & (dts <= np.datetime64(b))
            s = stat(recs[k], m)
            out['seg'][nm][sk] = s
            row += f'{s["CAGR"]:>13.2%}/{s["Sharpe"]:>7.2f}' if s else f'{"--":>22s}'
        print(row)

    for k in diag:
        v = np.array(diag[k], dtype=float)
        out['diag'][k] = float(np.nanmean(v))
    print(f'\n诊断: 条件数 样本{out["diag"]["cond_sam"]:.3g} / 联合{out["diag"]["cond_mgl"]:.3g}'
          f'   菲德勒选中率 样本{out["diag"]["fiedler_sam"]:.1%} / 联合{out["diag"]["fiedler_mgl"]:.1%}')
    out['nmi'] = out_nmi
    print(f'      相邻期聚类 NMI: 样本{out_nmi.get("sam", float("nan")):.3f} / '
          f'联合{out_nmi.get("mgl", float("nan")):.3f}')
    print(f'      行业内/行业间边权比 暴露图{out["diag"]["ind_ratio_mgl"]:.2f}× / '
          f'相关图{out["diag"]["ind_ratio_cor"]:.2f}×   平均度数{out["diag"]["avg_degree"]:.1f}')

    os.makedirs(f'{OUTPUT_DIR}/research', exist_ok=True)
    json.dump(out, open(f'{OUTPUT_DIR}/research/mingle_replicate.json', 'w'),
              ensure_ascii=False, indent=1, default=float)
    np.savez_compressed(f'{OUTPUT_DIR}/research/mingle_returns.npz',
                        dates=dts.astype('U10'), **{k: np.array(v) for k, v in recs.items()})
    print('\n已写 output/research/mingle_replicate.json')


if __name__ == '__main__':
    main()
