# -*- coding: utf-8 -*-
"""纸上交易账本: 上期结算 + 本期信号追加。

test 段已消耗, 这是唯一能继续积累无偏证据的方式——从今天起每期生成持仓、
事后对账, 半年后即有约 26 期真·实时样本外。

设计原则:
  1) **append-only**: 历史记录永不覆盖、永不回填。
  2) **单一打分实现**: 本脚本不含任何打分逻辑, 全部委托 generate_portfolio。
     研究期教训——同一策略存在多套打分实现时, 它们迟早会悄悄分叉。
  3) **配置冻结**: 打分侧从 config/frozen_alpha_v28.json 读取并记录其 SHA。

用法:
    python src/live/paper_ledger.py             # 结算上期 + 生成本期
    python src/live/paper_ledger.py --dry-run   # 只打印不写盘
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import PROJ_ROOT as PROJ, CACHE_DIR as CACHE, OUTPUT_DIR

import os, json, argparse, datetime
import numpy as np
from generate_portfolio import generate

LOG_DIR = f'{OUTPUT_DIR}/live'
LEDGER = f'{LOG_DIR}/paper_trading_ledger.jsonl'
STATE = f'{LOG_DIR}/ledger_state.json'


def settle(prev, dates, codes, logc, qz):
    """按上期信号日到当前的实际行情结算上一期。"""
    c2i = {c: i for i, c in enumerate(codes)}
    w0 = np.where(dates == prev['signal_date'])[0]
    if not len(w0):
        return None
    t0 = int(w0[0])
    t1 = min(t0 + 5, len(dates) - 1)
    if t1 <= t0:
        return None
    r, miss = 0.0, 0
    for c, w in prev['weights'].items():
        i = c2i.get(c)
        if i is None or not np.isfinite(logc[t1, i]) or not np.isfinite(logc[t0, i]):
            miss += 1
            continue
        r += w * (np.exp(logc[t1, i] - logc[t0, i]) - 1)
    b = float(np.prod(1 + qz[t0 + 1:t1 + 1]) - 1)
    return {'from': str(dates[t0]), 'to': str(dates[t1]),
            'port_ret': round(r, 6), 'bench_ret': round(b, 6),
            'excess': round((1 + r) / (1 + b) - 1, 6), 'n_missing': miss}


def main(dry_run=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    g = np.load(f'{CACHE}/daily_grid.npz')
    dates, codes, ret = g['dates'], g['codes'], g['ret']
    logc = np.cumsum(np.log1p(np.nan_to_num(ret, nan=0.0)), 0)
    qz = np.load(f'{CACHE}/quanzhi_ret.npy')

    prev = json.load(open(STATE)) if os.path.exists(STATE) else None
    settled = settle(prev, dates, codes, logc, qz) if prev else None
    if settled:
        print(f"[对账] {settled['from']} -> {settled['to']}  组合 {settled['port_ret']:+.2%}  "
              f"基准 {settled['bench_ret']:+.2%}  超额 {settled['excess']:+.2%}", flush=True)
    elif prev:
        print('[对账] 上期尚未满一个持有期, 跳过结算', flush=True)

    df, od, stats = generate(write=not dry_run)
    weights = {str(c): round(float(w), 6) for c, w in zip(df['code'], df['weight'])}
    rec = {'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'signal_date': stats['date'],
           'execute_hint': 'T+1: 11:30前卖出, 13:30-14:30买入(路径B)',
           'n_holdings': stats['n_hold'], 'turnover_oneside': stats['turnover_oneside'],
           'config_sha': stats['config_sha'], 'config_version': stats['config_version'],
           'settled_prev': settled, 'weights': weights}
    print(f"[信号] {rec['signal_date']}  持仓 {rec['n_holdings']} 只  "
          f"单边换手 {rec['turnover_oneside']:.1%}", flush=True)
    print('  前10大权重:', ', '.join(f'{c}:{v:.2%}' for c, v in
                                   sorted(weights.items(), key=lambda x: -x[1])[:10]), flush=True)
    if dry_run:
        print('\n[dry-run] 未写盘')
        return
    with open(LEDGER, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    json.dump(rec, open(STATE, 'w'), ensure_ascii=False, indent=1)
    print(f'\n已追加至 {LEDGER}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    main(ap.parse_args().dry_run)
