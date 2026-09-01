# -*- coding: utf-8 -*-
"""每周报告的可视化图表。

产出四张 PNG(附在邮件里, 不生成 html/json, 不上传任何在线服务):

  1. top50_scores    Top-50 逐股三块得分构成 + 代码与名称
  2. holdings        200 只目标持仓权重分布 + 权重最大的 40 只明细
  3. industry        行业权重 vs 活跃域基准, 标出 ±5% 偏离带
  4. ledger          纸上交易累计超额净值(账本满 2 期后才有内容)

只用 matplotlib, 中文走 Noto Sans CJK SC。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, CACHE_DIR as CACHE, OUTPUT_DIR, STOCK_ROOT

import os, json, glob, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130
plt.rcParams['savefig.bbox'] = 'tight'

C_ANCHOR, C_BEHAV, C_STRUCT = '#4C72B0', '#DD8452', '#55A868'
C_MAIN, C_GREY = '#4C72B0', '#B0B0B0'


def _names(codes):
    """代码 -> 中文简称。stock_info 是单个 parquet 文件, 不是按日期分片的目录。"""
    for cand in (f'{STOCK_ROOT}/info/stock_info.parquet',
                 f'{STOCK_ROOT}/info/security_info.parquet'):
        if not os.path.exists(cand):
            continue
        try:
            df = pd.read_parquet(cand)
        except Exception:
            continue
        cc = next((c for c in ('code', 'symbol', 'ts_code') if c in df.columns), None)
        nc = next((c for c in ('display_name', 'name', 'short_name') if c in df.columns), None)
        if cc and nc:
            m = dict(zip(df[cc].astype(str), df[nc].astype(str)))
            return {c: m.get(c, '') for c in codes}
    return {c: '' for c in codes}


def chart_top50(top, ds, out):
    d = top.head(50).iloc[::-1]          # 倒序: 排名第一画在最上
    nm = _names(list(d['code']))
    lab = [f"{r}. {c.split('.')[0]} {nm.get(c, '')}".strip()
           for r, c in zip(d['rank'], d['code'])]
    a = d['anchor'].values * 0.4
    b = d['behavior'].values * 0.4
    s = d['structure'].values * 0.2
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, 12))
    ax.barh(y, a, color=C_ANCHOR, label='五锚 (权重40%)')
    ax.barh(y, b, left=a, color=C_BEHAV, label='微观行为 (40%)')
    ax.barh(y, s, left=a + b, color=C_STRUCT, label='结构 (20%)')
    for i, v in enumerate(d['score'].values):
        ax.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=7.5, color='#333')
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlim(0, max(d['score'].max() * 1.13, 0.6))
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel('综合得分 (三块加权和)')
    ax.set_title(f'打分排名 Top-50   信号日 {ds}', fontsize=13, pad=26)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3,
              fontsize=9, frameon=False)
    ax.grid(axis='x', alpha=0.25, ls='--')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def chart_holdings(pf, ds, out):
    nm = _names(list(pf['code']))
    w = pf['weight'].values * 100
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.22)

    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(1, len(w) + 1), np.sort(w)[::-1], color=C_MAIN, lw=1.8)
    ax.fill_between(np.arange(1, len(w) + 1), np.sort(w)[::-1], color=C_MAIN, alpha=0.15)
    ax.axhline(1.0, color='#C44E52', ls='--', lw=1.2, label='个股上限 1%')
    ax.axhline(100 / len(w), color=C_GREY, ls=':', lw=1.2,
               label=f'等权基准 {100/len(w):.2f}%')
    ax.set_xlabel('按权重排序的持仓序号')
    ax.set_ylabel('权重 (%)')
    ax.set_title(f'{len(pf)} 只目标持仓的权重分布', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, ls='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(w.max() * 1.25, 1.1))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    d = pf.nlargest(40, 'weight').iloc[::-1]
    ax2 = fig.add_subplot(gs[1])
    y = np.arange(len(d))
    ax2.barh(y, d['weight'].values * 100, color=C_MAIN, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{c.split('.')[0]} {nm.get(c, '')}".strip() for c in d['code']],
                        fontsize=7.5)
    for i, v in enumerate(d['weight'].values * 100):
        ax2.text(v + 0.004, i, f'{v:.2f}%', va='center', fontsize=7, color='#333')
    ax2.set_xlabel('权重 (%)')
    ax2.set_title('权重最大的 40 只', fontsize=12)
    ax2.set_xlim(0, d['weight'].max() * 100 * 1.16)
    ax2.grid(axis='x', alpha=0.25, ls='--')
    ax2.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)

    fig.suptitle(f'目标持仓   信号日 {ds}', fontsize=13.5, y=0.985)
    fig.savefig(out)
    plt.close(fig)


def chart_industry(pf, ds, out, bench=None):
    """上栏: 持仓权重 vs 活跃域基准; 下栏: 偏离度。

    偏离度单独成栏, 而不是在上栏画一条 ±5% 的带——行业是类别轴, 跨类别的
    连续填充会把不相邻的行业连起来, 读起来像趋势, 实际没有这层含义。
    """
    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    imap = im[sorted(im)[-1]]
    pf = pf.copy()
    pf['ind'] = pf['code'].map(imap).fillna('未分类')
    g = pf.groupby('ind')['weight'].sum().sort_values(ascending=False)
    w = g.values * 100
    bv = np.array([bench.get(i, 0.0) for i in g.index]) * 100 if bench is not None else None
    x = np.arange(len(g))

    if bv is None:
        fig, ax = plt.subplots(figsize=(11, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.4), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})
    ax = axes[0]
    ax.bar(x, w, color=C_MAIN, alpha=0.85, label='持仓权重')
    if bv is not None:
        ax.plot(x, bv, 'o', color='#C44E52', ms=5, label='活跃域基准')
    ax.set_ylabel('权重 (%)')
    ax.set_title(f'行业分布   共 {len(g)} 个行业   信号日 {ds}', fontsize=12.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25, ls='--')
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    if bv is not None:
        dv = w - bv
        ax2 = axes[1]
        ax2.bar(x, dv, color=np.where(np.abs(dv) > 5, '#C44E52', C_GREY), alpha=0.9)
        ax2.axhline(0, color='#666', lw=0.9)
        for lv in (5, -5):
            ax2.axhline(lv, color='#C44E52', ls='--', lw=1.1)
        ax2.text(len(g) - 0.5, 5.15, '偏离上限 +5pp', fontsize=8,
                 color='#C44E52', ha='right', va='bottom')
        ax2.set_ylabel('偏离 (pp)')
        lim = max(6.2, np.abs(dv).max() * 1.25)
        ax2.set_ylim(-lim, lim)
        ax2.grid(axis='y', alpha=0.25, ls='--')
        ax2.set_axisbelow(True)
        ax2.set_xticks(x)
        ax2.set_xticklabels(g.index, rotation=45, ha='right', fontsize=8.5)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=45, ha='right', fontsize=8.5)

    for a_ in axes:
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def chart_ledger(out):
    f = f'{OUTPUT_DIR}/live/paper_trading_ledger.jsonl'
    if not os.path.exists(f):
        return False
    recs = [json.loads(l) for l in open(f, encoding='utf-8')]
    st = [(r['signal_date'], r['settled_prev']) for r in recs if r.get('settled_prev')]
    if len(st) < 2:
        return False
    ds = [s[1]['to'] for s in st]
    pr = np.array([s[1]['port_ret'] for s in st])
    bq = np.array([s[1]['bench_ret'] for s in st])
    ex = (1 + pr) / (1 + bq) - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
    a1.plot(ds, np.cumprod(1 + pr), 'o-', color=C_MAIN, ms=3.5, label='组合')
    a1.plot(ds, np.cumprod(1 + bq), 'o-', color=C_GREY, ms=3.5, label='中证全指')
    a1.plot(ds, np.cumprod(1 + ex), 'o-', color='#C44E52', ms=3.5, lw=2, label='超额')
    a1.axhline(1.0, color='#888', lw=0.8)
    a1.set_ylabel('净值')
    a1.set_title('纸上交易累计净值 (实时样本外)', fontsize=12.5)
    a1.legend(fontsize=9)
    a1.grid(alpha=0.25, ls='--')
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.3f}'))
    a2.bar(ds, ex * 100, color=np.where(ex >= 0, C_MAIN, '#C44E52'), alpha=0.85)
    a2.axhline(0, color='#888', lw=0.8)
    a2.set_ylabel('单期超额 (%)')
    a2.grid(axis='y', alpha=0.25, ls='--')
    for ax in (a1, a2):
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    plt.setp(a2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def build(ds=None, top_n=50):
    live = f'{OUTPUT_DIR}/live'
    if ds is None:
        cand = sorted(glob.glob(f'{live}/portfolio_*.csv'))
        if not cand:
            raise SystemExit('没有 portfolio_*.csv, 先跑 generate_portfolio.py')
        ds = os.path.basename(cand[-1])[10:-4]
    pf = pd.read_csv(f'{live}/portfolio_{ds}.csv')
    top = pd.read_csv(f'{live}/top{top_n}_{ds}.csv')

    # 活跃域的行业基准
    g = np.load(f'{CACHE}/daily_grid.npz')
    codes = g['codes']
    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    imap = im[sorted(im)[-1]]
    dom = np.array([imap.get(c, '') for c in codes])
    dom = dom[dom != '']
    bench = {u: float((dom == u).mean()) for u in set(dom)}

    cd = f'{live}/charts'
    os.makedirs(cd, exist_ok=True)
    out = []
    p = f'{cd}/top50_{ds}.png'; chart_top50(top, ds, p); out.append(p)
    p = f'{cd}/holdings_{ds}.png'; chart_holdings(pf, ds, p); out.append(p)
    p = f'{cd}/industry_{ds}.png'; chart_industry(pf, ds, p, bench); out.append(p)
    p = f'{cd}/ledger_{ds}.png'
    if chart_ledger(p):
        out.append(p)
    print('图表:', '  '.join(os.path.basename(x) for x in out))
    return ds, out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--asof', default=None)
    ap.add_argument('--top', type=int, default=50)
    a = ap.parse_args()
    build(a.asof, a.top)
