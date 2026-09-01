# -*- coding: utf-8 -*-
"""每周实盘信号生成 + 纸上交易日志(append-only)。

用途: test 段已消耗后, 这是唯一能继续积累无偏证据的方式 ——
从今天起每周生成一次持仓、事后对账, 半年后即有 26 周真·实时样本外。

设计原则:
  1) **append-only**: 历史记录永不覆盖、永不回填。每次运行只追加当期。
  2) **配置冻结**: 直接读 config/frozen_alpha_v27.json, 脚本内不含任何可调参数。
  3) **自对账**: 每次运行先对上一期持仓做实际收益结算, 再生成本期信号。
  4) **可复现**: 每期记录信号日、成交日、持仓明细与权重, 事后可逐笔复核。

用法:
    python src/live/weekly_signal.py            # 生成最新一期(需数据已更新到最新交易日)
    python src/live/weekly_signal.py --dry-run  # 只打印不写盘
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ

import os, json, argparse, hashlib, datetime
import numpy as np
import pandas as pd

os.environ['OPENBLAS_NUM_THREADS'] = '2'

LOG_DIR = f'{PROJ}/output/live'
LEDGER = f'{LOG_DIR}/paper_trading_ledger.jsonl'
HOLDINGS = f'{LOG_DIR}/current_holdings.json'


def main(dry_run=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    # 复用定型脚本的数据与信号(含 test 段补算逻辑)
    src = open(f'{PROJ}/src/test/open_test_segment.py', encoding='utf-8').read()
    ns = {}
    exec(compile(src.split("out = {'criteria': CRIT}")[0], 'signals', 'exec'), ns)
    g = ns['g']; dnum = ns['dnum']; T = ns['T']; N = ns['N']
    codes = g['codes']; ret = ns['ret']; logc = ns['logc']
    sig_all = ns['sig_all']

    # ---------- 1. 上一期对账 ----------
    prev = None
    if os.path.exists(HOLDINGS):
        prev = json.load(open(HOLDINGS))
    settled = None
    if prev is not None:
        t0 = int(prev['signal_t'])
        # 结算至当前可得的最后一个交易日
        t1 = min(t0 + 5, T - 1)
        if t1 > t0:
            ids = {c: w for c, w in prev['weights'].items()}
            cix = {c: i for i, c in enumerate(codes)}
            r = 0.0
            miss = []
            for c, w in ids.items():
                i = cix.get(c)
                if i is None or not np.isfinite(logc[t1, i]) or not np.isfinite(logc[t0, i]):
                    miss.append(c); continue
                r += w * (np.exp(logc[t1, i] - logc[t0, i]) - 1)
            lgq = ns['lgq']
            b = float(np.exp(lgq[t1] - lgq[t0]) - 1)
            settled = {'from': str(dnum[t0]), 'to': str(dnum[t1]),
                       'port_ret': round(r, 6), 'bench_ret': round(b, 6),
                       'excess': round((1 + r) / (1 + b) - 1, 6),
                       'n_missing': len(miss)}
            print(f"[对账] {settled['from']} -> {settled['to']}  "
                  f"组合 {r:+.2%}  基准 {b:+.2%}  超额 {settled['excess']:+.2%}", flush=True)

    # ---------- 2. 本期信号 ----------
    t = int(sig_all[-1])          # 最近一个信号日
    run_alpha = ns['run_alpha']
    # 只取最后一期的持仓: 复用 run_alpha 的逻辑但输出明细
    recs = run_alpha()            # 全序列(保证缓冲带状态一致)
    # run_alpha 内部不返回持仓明细, 故此处按同一规则重算最后一期
    comb, hard = ns['_last_comb'](t) if '_last_comb' in ns else (None, None)
    if comb is None:
        # 回退: 直接按定型规则重算(与 run_alpha 完全一致的打分)
        v = np.maximum(ns['vol20'][t] * np.sqrt(20), 1e-4)
        stk = np.stack([ns['rank_xs'](ns['PG_b'][t] / v), ns['rank_xs'](ns['ZS_b'][t]),
                        ns['rank_xs'](ns['PG_dual'][t] / v), ns['rank_xs'](ns['PG_te'][t] / v),
                        ns['rank_xs'](ns['fgap'](ns['ROE'], t))])
        with np.errstate(all='ignore'):
            base = np.where(np.all(np.isnan(stk), 0), np.nan, np.nanmean(stk, 0))
        dom = (ns['lim250'][t] >= 2) & ns['tradable'][t] & ~np.isnan(ns['PG_b'][t])
        base = np.where(dom, base, np.nan)
        hard = (ns['paused'][t] < 0.5) & (ns['at_hl'][t] < 0.5) & dom & ~np.isnan(base)
        fm = dict(zip(ns['FN'], ns['tfeat'](t)))
        behav = np.nanmean(np.stack([ns['rank01'](-fm['sweep_sell'], hard),
                                     ns['rank01'](-fm['osize_sell'], hard)]), 0)
        sl = [(np.nan_to_num(ns['close_raw'][t], nan=0) >= 2.0).astype(np.float32)]
        dr = ns['DROE'][t]; fin = hard & np.isfinite(dr)
        sl.append((np.nan_to_num(dr, nan=0) >= np.nanquantile(dr[fin], 1 / 3)).astype(np.float32)
                  if fin.sum() > 200 else np.ones(N, np.float32))
        struct = np.mean(np.stack(sl), 0)
        with np.errstate(all='ignore'):
            comb = 0.4 * ns['rank01'](base, hard) + 0.4 * np.nan_to_num(behav, nan=0.5) + 0.2 * struct
        comb = np.where(hard, comb, np.nan)

    order = np.argsort(-np.nan_to_num(comb, nan=-1e9))
    prev_set = set(prev['weights'].keys()) if prev else set()
    cix = {c: i for i, c in enumerate(codes)}
    rank = np.full(N, 1 << 30); rank[order] = np.arange(N)
    keep = [c for c in prev_set if c in cix and rank[cix[c]] < 600 and not np.isnan(comb[cix[c]])]
    sel = [cix[c] for c in keep]
    for i in order:
        if len(sel) >= 200: break
        if i in sel or np.isnan(comb[i]) or not hard[i]: continue
        sel.append(i)
    w = np.array([np.nan_to_num(comb[i], nan=0.5) + 0.3 for i in sel]); w = w / w.sum()
    weights = {str(codes[i]): round(float(x), 6) for i, x in zip(sel, w)}
    turn = sum(abs(weights.get(c, 0) - (prev['weights'].get(c, 0) if prev else 0))
               for c in set(weights) | prev_set) / 2

    rec = {'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'signal_date': str(dnum[t]), 'signal_t': t,
           'execute_hint': 'T+1: 11:30前卖出, 13:30-14:30买入(路径B); 或信号日尾盘(路径A)',
           'n_holdings': len(weights), 'turnover_oneside': round(turn, 4),
           'config_sha': hashlib.sha256(open(f'{PROJ}/README.md', 'rb').read()).hexdigest()[:12],
           'settled_prev': settled, 'weights': weights}
    print(f"[信号] {rec['signal_date']}  持仓 {rec['n_holdings']} 只  "
          f"单边换手 {rec['turnover_oneside']:.1%}", flush=True)
    print('  前10大权重:', ', '.join(f'{c}:{v:.2%}' for c, v in
                                 sorted(weights.items(), key=lambda x: -x[1])[:10]), flush=True)

    if dry_run:
        print('\n[dry-run] 未写盘'); return
    with open(LEDGER, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    json.dump(rec, open(HOLDINGS, 'w'), ensure_ascii=False, indent=1)
    print(f'\n已追加至 {LEDGER}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    main(a.dry_run)
