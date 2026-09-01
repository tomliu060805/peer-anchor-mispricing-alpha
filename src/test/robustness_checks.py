# -*- coding: utf-8 -*-
"""稳健性三检验(不增加任何策略参数, 仅增强证据强度):

  1) 随机组合零基准: 同域内随机挑 N 只、同周频换、同成本, 跑 M 次,
     看 v27 超额落在零基准分布的第几个标准差
  2) Block bootstrap 置信区间: 块长保留自相关, 给出 IR/超额年化的 90% CI
  3) 滚动 12 个月 IR 时间序列: 展示有效性是持续的还是靠某几段撑起来

三段(dev/val/test)分别报告。test 段仅做展示, 不用于任何选择。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ

import os, json
import numpy as np
import pandas as pd

os.environ['OPENBLAS_NUM_THREADS'] = '2'

# 复用开封脚本的数据装载 + 信号补算 + run_alpha
_src = open(f'{PROJ}/src/test/open_test_segment.py', encoding='utf-8').read()
exec(_src.split("out = {'criteria': CRIT}")[0])

SEG = {'dev': ('2016-01-01', '2022-12-31'),
       'val': ('2023-01-01', '2024-08-16'),
       'test': ('2024-08-19', '2026-12-31')}
NHOLD, BUF, M_SIM = 200, 600, 500
RNG = np.random.default_rng(20260828)


def seg_mask(ds, k):
    a, b = SEG[k]
    return (ds >= np.datetime64(a)) & (ds <= np.datetime64(b))


def perf(pr, bq, per_year=252 / 5):
    ex = (1 + pr) / (1 + bq) - 1
    nav = np.cumprod(1 + ex)
    return dict(ExAnn=float(nav[-1] ** (per_year / len(ex)) - 1),
                IR=float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(per_year)))


# ---------------- 策略基准线 ----------------
recs = run_alpha()
ds = np.array([r[0] for r in recs])
pr = np.array([r[1] for r in recs])
bq = np.array([r[2] for r in recs])
print(f'策略序列: {ds[0]} ~ {ds[-1]}  {len(ds)} 期', flush=True)

out = {'config': 'v27 路径B, N=200, 缓冲带600, 双边20bp', 'n_sim': M_SIM}
out['strategy'] = {k: perf(pr[seg_mask(ds, k)], bq[seg_mask(ds, k)]) for k in SEG}
for k, v in out['strategy'].items():
    print(f"  策略 {k}: ExAnn={v['ExAnn']:.2%} IR={v['IR']:.2f}", flush=True)


# ---------------- 1) 随机组合零基准 ----------------
# 同一活跃域、同样 N=200、同样缓冲带换手规则, 唯一区别是打分换成随机数
def run_random(seed):
    rng = np.random.default_rng(seed)
    holdings = {}
    rr = []
    for si, t in enumerate(sig_all):
        if si + 1 >= len(sig_all) or t < 8:
            continue
        dom = (lim250[t] >= 2) & tradable[t] & ~np.isnan(PG_b[t])
        hard = (paused[t] < 0.5) & (at_hl[t] < 0.5) & dom
        if hard.sum() < 100:
            holdings = {}
            continue
        score = np.where(hard, rng.random(N), np.nan)   # 唯一改动: 随机打分
        order = np.argsort(-np.nan_to_num(score, nan=-1e9))
        rank = np.full(N, 1 << 30)
        rank[order] = np.arange(N)
        can_sell = (paused[t] < 0.5) & (at_ll[t] < 0.5)
        new_h = dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2] >= BUF) or np.isnan(score[i2])) and can_sell[i2]:
                del new_h[i2]
        for i2 in order:
            if len(new_h) >= NHOLD:
                break
            if i2 in new_h or np.isnan(score[i2]) or not hard[i2]:
                continue
            new_h[i2] = 0.0
        if len(new_h) < 40:
            holdings = {}
            continue
        w = np.array([np.nan_to_num(score[i2], nan=0.5) + 0.3 for i2 in new_h])
        w = w / w.sum()
        wt = dict(zip(new_h, w))
        turn = sum(abs(wt.get(i2, 0) - holdings.get(i2, 0)) for i2 in set(wt) | set(holdings))
        t_next = sig_all[si + 1]
        p = sum(ww * (np.exp(logc[t_next, i2] - logc[t, i2]) - 1) for i2, ww in wt.items())
        p -= turn * COST
        rr.append((dnum[t], p, np.exp(lgq[t_next] - lgq[t]) - 1))
        holdings = wt
    return np.array([r[0] for r in rr]), np.array([r[1] for r in rr]), np.array([r[2] for r in rr])


print(f'\n随机组合零基准 ({M_SIM} 次)...', flush=True)
sims = {k: {'ExAnn': [], 'IR': []} for k in SEG}
for s in range(M_SIM):
    rds, rpr, rbq = run_random(10000 + s)
    for k in SEG:
        m = seg_mask(rds, k)
        if m.sum() < 10:
            continue
        d = perf(rpr[m], rbq[m])
        sims[k]['ExAnn'].append(d['ExAnn'])
        sims[k]['IR'].append(d['IR'])
    if (s + 1) % 100 == 0:
        print(f'  {s + 1}/{M_SIM}', flush=True)

out['random_baseline'] = {}
for k in SEG:
    a = np.array(sims[k]['ExAnn'])
    i = np.array(sims[k]['IR'])
    sv = out['strategy'][k]
    out['random_baseline'][k] = {
        'rand_ExAnn_mean': round(float(a.mean()), 4), 'rand_ExAnn_std': round(float(a.std()), 4),
        'rand_IR_mean': round(float(i.mean()), 3), 'rand_IR_std': round(float(i.std()), 3),
        'z_ExAnn': round(float((sv['ExAnn'] - a.mean()) / (a.std() + 1e-12)), 2),
        'z_IR': round(float((sv['IR'] - i.mean()) / (i.std() + 1e-12)), 2),
        'pct_beaten': round(float((a < sv['ExAnn']).mean()), 4)}
    r = out['random_baseline'][k]
    print(f"  {k}: 随机均值 {r['rand_ExAnn_mean']:.2%}±{r['rand_ExAnn_std']:.2%} | "
          f"策略 {sv['ExAnn']:.2%} | z={r['z_ExAnn']} | 超越 {r['pct_beaten']:.1%}", flush=True)


# ---------------- 2) Block bootstrap 置信区间 ----------------
def block_boot(pr_, bq_, block=6, n=2000, per_year=252 / 5):
    ex = (1 + pr_) / (1 + bq_) - 1
    L = len(ex)
    nb = int(np.ceil(L / block))
    stats = {'ExAnn': [], 'IR': []}
    for _ in range(n):
        st = RNG.integers(0, max(L - block, 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, L)) for s in st])[:L]
        e = ex[idx]
        nav = np.cumprod(1 + e)
        stats['ExAnn'].append(float(nav[-1] ** (per_year / len(e)) - 1))
        stats['IR'].append(float(e.mean() / (e.std() + 1e-12) * np.sqrt(per_year)))
    return {kk: {'p05': round(float(np.percentile(v, 5)), 4),
                 'p50': round(float(np.percentile(v, 50)), 4),
                 'p95': round(float(np.percentile(v, 95)), 4)} for kk, v in stats.items()}


print('\nBlock bootstrap 90% 置信区间 (块长6周)...', flush=True)
out['bootstrap'] = {}
for k in SEG:
    m = seg_mask(ds, k)
    if m.sum() < 20:
        continue
    out['bootstrap'][k] = block_boot(pr[m], bq[m])
    b = out['bootstrap'][k]
    print(f"  {k}: ExAnn [{b['ExAnn']['p05']:.2%}, {b['ExAnn']['p95']:.2%}] | "
          f"IR [{b['IR']['p05']:.2f}, {b['IR']['p95']:.2f}]", flush=True)


# ---------------- 3) 滚动 12 个月 IR ----------------
ex_all = (1 + pr) / (1 + bq) - 1
W = 52
roll = []
for i in range(W - 1, len(ex_all)):
    e = ex_all[i - W + 1:i + 1]
    roll.append((str(ds[i]), float(e.mean() / (e.std() + 1e-12) * np.sqrt(252 / 5)),
                 float(np.prod(1 + e) - 1)))
out['rolling_12m'] = {'dates': [r[0] for r in roll],
                      'IR': [round(r[1], 3) for r in roll],
                      'ExRet': [round(r[2], 4) for r in roll]}
ir = np.array([r[1] for r in roll])
out['rolling_summary'] = {'n_windows': len(ir), 'IR_mean': round(float(ir.mean()), 2),
                          'IR_min': round(float(ir.min()), 2), 'IR_max': round(float(ir.max()), 2),
                          'pct_IR_gt_0': round(float((ir > 0).mean()), 3),
                          'pct_IR_gt_0.5': round(float((ir > 0.5).mean()), 3)}
print(f"\n滚动12月IR: 均值{out['rolling_summary']['IR_mean']} "
      f"区间[{out['rolling_summary']['IR_min']}, {out['rolling_summary']['IR_max']}] "
      f"IR>0占比{out['rolling_summary']['pct_IR_gt_0']:.1%} "
      f"IR>0.5占比{out['rolling_summary']['pct_IR_gt_0.5']:.1%}", flush=True)

os.makedirs(f'{PROJ}/output/test', exist_ok=True)
json.dump(out, open(f'{PROJ}/output/test/robustness.json', 'w'), ensure_ascii=False, indent=1)
np.savez(f'{PROJ}/output/test/robustness_series.npz',
         dates=ds.astype(str), pr=pr, bq=bq,
         roll_dates=np.array([r[0] for r in roll]), roll_ir=ir)
print('\nsaved output/test/robustness.json')
