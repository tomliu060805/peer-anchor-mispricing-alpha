# -*- coding: utf-8 -*-
"""财务比率面板: 六类结构指标, PIT 对齐。

背景
----
一篇用标普500检验"行业分类能否解释财务结构"的工作提出: 从财务比率反推行业标签
的监督学习准确率有限, 而无监督聚类会得到跨越行业边界的财务结构分组。
本脚本先把同样口径的比率在 A 股上构建出来, 后续脚本据此做复现与检验。

六类(与该工作对齐, 用三张报表自行计算, 不依赖预算好的指标):
  盈利      ROE / ROA / 毛利率 / 净利率
  杠杆      资产负债率 / 有息负债率 / 权益乘数
  流动性    流动比率 / 速动比率 / 现金比率
  利息保障  EBIT / 利息支出
  效率      总资产周转 / 应收周转 / 存货周转
  现金生成  经营现金流/营收 / 经营现金流/总资产 / 经营现金流/净利润

PIT
---
每日 parquet 已是"当日可得的最新报告", 且 pub_date 全部早于当日。
仍按 fi_grid 的口径再滞后一日 (pub_date <= t-1) 作为保险。

产出 cache/finratio_grid.npz: ratios (T, R, N) float32 + names
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, STOCK_ROOT

import os, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

N_WORKERS = int(os.environ.get('N_WORKERS', '80'))
FUND = f'{STOCK_ROOT}/fundamental'

BS = ['total_assets', 'total_liability', 'total_owner_equities', 'total_current_assets',
      'total_current_liability', 'inventories', 'cash_equivalents', 'account_receivable',
      'shortterm_loan', 'longterm_loan', 'bonds_payable',
      'non_current_liability_in_one_year']
IS = ['total_operating_revenue', 'operating_revenue', 'operating_cost', 'total_profit',
      'financial_expense', 'net_profit', 'np_parent_company_owners', 'interest_expense']
CF = ['net_operate_cash_flow']

NAMES = ['roe', 'roa', 'gross_margin', 'net_margin',
         'debt_ratio', 'ib_debt_ratio', 'equity_mult',
         'current_ratio', 'quick_ratio', 'cash_ratio',
         'int_cover',
         'asset_turn', 'recv_turn', 'inv_turn',
         'ocf_rev', 'ocf_asset', 'ocf_np']
_CTX = {}


def _init(dates, codes):
    _CTX['dates'] = dates
    _CTX['cix'] = {c: i for i, c in enumerate(codes)}
    _CTX['N'] = len(codes)


def _safe(a, b, lo=None, hi=None):
    """a/b, 分母无效或过小则 NaN; 可选温和截断以防极值主导标准化。"""
    b = np.asarray(b, np.float64)
    out = np.where(np.abs(b) > 1e-6, np.asarray(a, np.float64) / np.where(b == 0, np.nan, b), np.nan)
    if lo is not None:
        out = np.clip(out, lo, hi)
    return out


def _load(i):
    dates, cix, Nn = _CTX['dates'], _CTX['cix'], _CTX['N']
    ds = str(dates[i])
    out = np.full((len(NAMES), Nn), np.nan, np.float32)
    try:
        b = pd.read_parquet(f'{FUND}/balance_sheet/{ds}.parquet',
                            columns=['code', 'pub_date'] + BS)
        s = pd.read_parquet(f'{FUND}/income_statement/{ds}.parquet',
                            columns=['code', 'pub_date'] + IS)
        c = pd.read_parquet(f'{FUND}/cash_flow_statement/{ds}.parquet',
                            columns=['code', 'pub_date'] + CF)
    except Exception:
        return i, out
    cut = pd.Timestamp(ds) - pd.Timedelta(days=1)
    for d in (b, s, c):
        d.drop(d.index[pd.to_datetime(d['pub_date'], errors='coerce') > cut], inplace=True)
    d = b.merge(s, on='code', suffixes=('', '_is')).merge(c, on='code', suffixes=('', '_cf'))
    if len(d) == 0:
        return i, out
    g = lambda k: pd.to_numeric(d[k], errors='coerce').values.astype(np.float64)

    TA, TL, TE_ = g('total_assets'), g('total_liability'), g('total_owner_equities')
    CA, CL, INV = g('total_current_assets'), g('total_current_liability'), g('inventories')
    CASH, AR = g('cash_equivalents'), g('account_receivable')
    IB = (np.nan_to_num(g('shortterm_loan')) + np.nan_to_num(g('longterm_loan'))
          + np.nan_to_num(g('bonds_payable')) + np.nan_to_num(g('non_current_liability_in_one_year')))
    REV, OC = g('total_operating_revenue'), g('operating_cost')
    TP, FE, NP = g('total_profit'), g('financial_expense'), g('np_parent_company_owners')
    IE = g('interest_expense')
    OCF = g('net_operate_cash_flow')
    # EBIT ≈ 利润总额 + 财务费用(含利息); 利息支出优先用 interest_expense, 缺失回落财务费用
    EBIT = TP + np.nan_to_num(FE)
    INT = np.where(np.isfinite(IE) & (IE > 0), IE, np.where(FE > 0, FE, np.nan))

    vals = [
        _safe(NP, TE_, -5, 5), _safe(NP, TA, -2, 2),
        _safe(REV - OC, REV, -5, 5), _safe(NP, REV, -5, 5),
        _safe(TL, TA, 0, 3), _safe(IB, TA, 0, 3), _safe(TA, TE_, -50, 50),
        _safe(CA, CL, 0, 50), _safe(CA - np.nan_to_num(INV), CL, 0, 50), _safe(CASH, CL, 0, 50),
        _safe(EBIT, INT, -100, 100),
        _safe(REV, TA, 0, 20), _safe(REV, AR, 0, 500), _safe(OC, INV, 0, 500),
        _safe(OCF, REV, -20, 20), _safe(OCF, TA, -5, 5), _safe(OCF, NP, -50, 50),
    ]
    ix = np.array([cix.get(x, -1) for x in d['code']])
    ok = ix >= 0
    for j, v in enumerate(vals):
        out[j, ix[ok]] = v[ok].astype(np.float32)
    return i, out


def main():
    g = np.load(f'{CACHE}/daily_grid.npz')
    dates, codes = g['dates'], g['codes']
    _init(dates, codes)
    t0 = time.time()
    print(f'构建财务比率面板: {len(dates)} 日 x {len(NAMES)} 比率 x {len(codes)} 股', flush=True)
    with ProcessPoolExecutor(N_WORKERS) as ex:
        outs = list(ex.map(_load, range(len(dates)), chunksize=8))
    R = np.full((len(dates), len(NAMES), len(codes)), np.nan, np.float32)
    for i, o in outs:
        R[i] = o
    np.savez_compressed(f'{CACHE}/finratio_grid.npz', ratios=R,
                        names=np.array(NAMES, 'U16'), dates=dates, codes=codes)
    print(f'完成 {R.shape} ({time.time()-t0:.0f}s)', flush=True)
    for j, n in enumerate(NAMES):
        print(f'  {n:14s} 末行非空 {np.isfinite(R[-1, j]).mean():.3f}  '
              f'中位 {np.nanmedian(R[-1, j]):>10.3f}')


if __name__ == '__main__':
    main()
