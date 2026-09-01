# -*- coding: utf-8 -*-
"""每周打分排名 Top-50 报告。

与 generate_portfolio 的区别: 后者出的是**可执行的目标持仓**(含缓冲带、
个股上限、行业偏离带, 会保留上期股票), 这里出的是**当期纯打分排名**——
不受上期持仓影响, 用于看当周信号本身指向哪些票。

两者不会完全一致, 这是设计使然:
  - Top-50 是"这周分数最高的 50 只"
  - 持仓是"综合换手成本与约束后, 这周该持有的 200 只"

产出:
  output/live/top50_{date}.csv    排名/代码/总分/三块分项/行业/是否已在持仓
  output/live/top50_{date}.txt    可直接阅读的文本版

用法:
    python src/live/weekly_top50.py            # 最新交易日
    python src/live/weekly_top50.py --asof 2026-08-31
    python src/live/weekly_top50.py --top 100
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import PROJ_ROOT as PROJ, CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, pickle, argparse, datetime
import numpy as np
import pandas as pd
import generate_portfolio as gp


def report(asof=None, top=50, write=True):
    # score_all=True 让 generate 额外返回全市场分数向量, 不受持仓/约束影响
    df_hold, _, stats, extra = gp.generate(asof=asof, write=False, return_scores=True)
    comb, hard, codes, log_date = extra['comb'], extra['hard'], extra['codes'], stats['date']
    anchor, behav, struct = extra['anchor'], extra['behav'], extra['struct']

    ok = np.where(hard & ~np.isnan(comb))[0]
    order = ok[np.argsort(-comb[ok])][:top]

    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    g = np.load(f'{CACHE}/daily_grid.npz')
    dates, close = g['dates'], g['close']
    t = int(np.where(dates == log_date)[0][0])
    mk = [m for m in sorted(im) if m <= str(dates[t - 1])]
    imap = im[mk[-1]] if mk else {}

    held = set(df_hold['code'])
    d = pd.DataFrame({
        'rank': np.arange(1, len(order) + 1),
        'code': codes[order],
        'industry': [imap.get(c, '') for c in codes[order]],
        'close': np.round(close[t, order].astype(np.float64), 2),
        'score': np.round(comb[order].astype(np.float64), 4),
        'anchor': np.round(anchor[order].astype(np.float64), 3),
        'behavior': np.round(np.nan_to_num(behav[order], nan=0.5).astype(np.float64), 3),
        'structure': np.round(struct[order].astype(np.float64), 3),
        'in_portfolio': ['是' if c in held else '' for c in codes[order]],
    })

    hdr = (f'打分排名 Top-{top}   信号日 {log_date}\n'
           f'候选池 {stats["candidates"]} 只 (活跃域 {stats["domain_size"]})   '
           f'配置 {stats["config_version"]}  SHA {stats["config_sha"]}\n'
           f'分数 = 五锚 40% + 微观行为 40% + 结构 20%,  三项均为域内分位(0~1)\n'
           + '=' * 96)
    body = d.to_string(index=False)
    tail = ('\n注: 本表是当周纯打分排名, 不等于建议持仓——持仓另受缓冲带(排名跌出600才卖)、\n'
            '    个股1%上限与行业±5%偏离带约束。执行按路径B: T+1 11:30前卖, 13:30-14:30买。')
    print(hdr); print(body); print(tail)

    if write:
        od = f'{OUTPUT_DIR}/live'
        os.makedirs(od, exist_ok=True)
        d.to_csv(f'{od}/top{top}_{log_date}.csv', index=False)
        with open(f'{od}/top{top}_{log_date}.txt', 'w', encoding='utf-8') as f:
            f.write(hdr + '\n' + body + '\n' + tail + '\n')
        print(f'\n已写 output/live/top{top}_{log_date}.csv / .txt')
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--asof', default=None)
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    report(asof=a.asof, top=a.top, write=not a.dry_run)
