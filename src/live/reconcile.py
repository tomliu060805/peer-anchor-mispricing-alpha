# -*- coding: utf-8 -*-
"""生产打分 vs 回测打分 对拍。

生产脚本 generate_portfolio.py 是对研究代码的**重新实现**——为了摆脱
"跑某个研究脚本的副作用"这种依赖。重新实现就有偏差风险：如果打分逻辑
与回测有出入，产出的持仓就不是被验证过的那个策略。

本脚本在同一批网络输入下，逐日比较两条路径选出的股票集合。
判据：交集/并集 >= 0.98 且分数相关 >= 0.999。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ

import os, json
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np

_src = open(f'{PROJ}/src/test/open_test_segment.py', encoding='utf-8').read()
exec(_src.split("out = {'criteria': CRIT}")[0])

_sys.path.insert(0, f'{PROJ}/src/live')
import generate_portfolio as gp

# **必须取回测信号网格上的日期**。回测的 PG_b / ZS_b / PG_dual / PG_te 只在
# sig_all 的日期上被计算, 其余日期是 NaN 或上一期的陈旧值。在非网格日期上比较,
# 等于拿生产的现算值去比回测的空值——会得到"低于随机重叠"的假判负。
# (第一版对拍就栽在这里: 5 个日期只有 1 个在网格上, 其余全塌。)
DATES = [str(dates[t]) for t in sig_all[-40::8]]
rows = []
for ds in DATES:
    w = np.where(dates == ds)[0]
    if not len(w):
        print(f'{ds}: 非交易日, 跳过')
        continue
    t = int(w[0])

    # ---- 回测路径 ----
    v = np.maximum(vol20[t] * np.sqrt(20), 1e-4)
    stk = np.stack([rank_xs(PG_b[t] / v), rank_xs(ZS_b[t]), rank_xs(PG_dual[t] / v),
                    rank_xs(PG_te[t] / v), rank_xs(fgap(ROE, t))])
    with np.errstate(all='ignore'):
        base = np.where(np.all(np.isnan(stk), 0), np.nan, np.nanmean(stk, 0))
    dom = (lim250[t] >= 2) & tradable[t] & ~np.isnan(PG_b[t])
    base = np.where(dom, base, np.nan)
    hard = (paused[t] < 0.5) & (at_hl[t] < 0.5) & dom & ~np.isnan(base)
    fm = dict(zip(FN, tfeat(t)))
    behav_r = np.nanmean(np.stack([rank01(-fm['sweep_sell'], hard),
                                   rank01(-fm['osize_sell'], hard)]), 0)
    sl = [(np.nan_to_num(close_raw[t], nan=0) >= 2.0).astype(np.float32)]
    dr = DROE[t]
    fin = hard & np.isfinite(dr)
    sl.append((np.nan_to_num(dr, nan=0) >= np.nanquantile(dr[fin], 1 / 3)).astype(np.float32)
              if fin.sum() > 200 else np.ones(N, np.float32))
    struct_r = np.mean(np.stack(sl), 0)
    with np.errstate(all='ignore'):
        comb_bt = 0.4 * rank01(base, hard) + 0.4 * np.nan_to_num(behav_r, nan=0.5) + 0.2 * struct_r
    comb_bt = np.where(hard, comb_bt, np.nan)
    top_bt = set(np.argsort(-np.nan_to_num(comb_bt, nan=-1e9))[:200])

    # ---- 生产路径 ----
    lh = f'{PROJ}/output/live/last_holdings.json'
    bak = lh + '.bak'
    if os.path.exists(lh):
        os.rename(lh, bak)                       # 对拍要求空仓起步(去掉缓冲带影响)
    try:
        df, _, _ = gp.generate(asof=ds, write=False)
    finally:
        if os.path.exists(bak):
            os.rename(bak, lh)
    c2i = {c: i for i, c in enumerate(codes)}
    top_pr = set(c2i[c] for c in df['code'] if c in c2i)

    inter = len(top_bt & top_pr)
    union = len(top_bt | top_pr)
    jac = inter / max(union, 1)
    both = np.array(sorted(top_bt & top_pr))
    corr = float(np.corrcoef(comb_bt[both], [float(df.set_index('code').loc[codes[i], 'score'])
                                             for i in both])[0, 1]) if len(both) > 10 else np.nan
    rows.append({'date': ds, 'inter': inter, 'jaccard': round(jac, 4),
                 'score_corr': round(corr, 6),
                 'pass': bool(jac >= 0.98 and corr >= 0.999)})
    print(f'{ds}  交集 {inter}/200  Jaccard {jac:.4f}  分数相关 {corr:.6f}  '
          f'{"通过" if rows[-1]["pass"] else "不通过"}', flush=True)

ok = all(r['pass'] for r in rows) and len(rows) > 0
print('\n对拍结论:', '生产实现与回测一致' if ok else '存在偏差, 需排查')
json.dump({'rows': rows, 'pass': ok},
          open(f'{PROJ}/output/live/reconcile.json', 'w'), ensure_ascii=False, indent=1)
