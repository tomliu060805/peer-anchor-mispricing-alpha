# -*- coding: utf-8 -*-
"""系统性成分复现的两张图。

  1. 行业一致性 vs 稀疏度 —— 两条曲线都随稀疏度陡升, 说明原对照被稀疏度混淆
  2. 各方法净值 + 随机图零基准带
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(_HERE, '..'))
from paths import OUTPUT_DIR, CHART_DIR

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
C_EXP, C_COR, C_MGL, C_EQ, C_MV, C_RAND = '#4C72B0', '#C44E52', '#4C72B0', '#8C8C8C', '#55A868', '#DD8452'


def chart_sparsity(out):
    j = json.load(open(f'{OUTPUT_DIR}/research/mingle_industry_sparsity.json'))
    r = j['rows']
    deg = np.array([x['avg_degree'] for x in r])
    ex = np.array([x['ratio_exposure'] for x in r])
    co = np.array([x['ratio_corr_matched'] for x in r])
    o = np.argsort(deg)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'wspace': 0.24})

    a1.plot(deg[o], ex[o], 'o-', color=C_EXP, lw=2, ms=6, label='暴露相似图')
    a1.plot(deg[o], co[o], 's-', color=C_COR, lw=2, ms=6, label='收益相关图（截断到等边数）')
    a1.axhline(5.30, color='#666', ls='--', lw=1.2)
    a1.text(deg.min() * 1.05, 5.45, '原工作报告的暴露图 5.30×', fontsize=8.5,
            ha='left', color='#666')
    a1.axhline(2.42, color='#999', ls=':', lw=1.2)
    a1.text(deg.min() * 1.05, 2.52, '原工作报告的相关图 2.42×', fontsize=8.5,
            ha='left', color='#999')
    a1.axvspan(7, 15, color='#4C72B0', alpha=0.09)
    a1.set_xscale('log')
    a1.set_ylim(0.9, ex.max() * 1.12)
    a1.text(10.2, 1.05, '可用于聚类的密度', fontsize=8.5, ha='center', color='#4C72B0')
    a1.set_xlabel('平均度数（对数轴，越左越稀疏）')
    a1.set_ylabel('行业内 / 行业间 平均边权比')
    a1.set_title('行业一致性是稀疏度的函数', fontsize=12.5)
    a1.legend(fontsize=9, loc='center right', framealpha=0.95)
    a1.grid(alpha=0.25, ls='--')
    a1.set_axisbelow(True)
    for sp in ('top', 'right'):
        a1.spines[sp].set_visible(False)

    rel = ex / co
    a2.plot(deg[o], rel[o], 'o-', color='#7B4EA8', lw=2, ms=6)
    a2.axhline(1.0, color='#333', lw=1.2)
    a2.fill_between(deg[o], 1.0, rel[o], where=(rel[o] < 1), color='#C44E52', alpha=0.13)
    a2.fill_between(deg[o], 1.0, rel[o], where=(rel[o] >= 1), color='#55A868', alpha=0.13)
    a2.axvspan(7, 15, color='#4C72B0', alpha=0.09)
    a2.set_xscale('log')
    a2.text(10.2, rel.min() + (rel.max() - rel.min()) * 0.06, '可用于聚类的密度',
            fontsize=8.5, ha='center', color='#4C72B0')
    a2.set_xlabel('平均度数（对数轴）')
    a2.set_ylabel('暴露图 ÷ 相关图（等边数）')
    a2.set_title('等边数对照：可用密度上暴露图一致更差', fontsize=12.5)
    a2.grid(alpha=0.25, ls='--')
    a2.set_axisbelow(True)
    for sp in ('top', 'right'):
        a2.spines[sp].set_visible(False)

    fig.suptitle('等稀疏度对照推翻"暴露图行业一致性更高"的结论', fontsize=13.5, y=0.99)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def chart_nav(out):
    z = np.load(f'{OUTPUT_DIR}/research/mingle_returns.npz')
    dts = z['dates'].astype('datetime64[D]')
    fig, ax = plt.subplots(figsize=(12, 6))
    rand = np.array([np.cumprod(1 + z[f'w_cc_rand{r}'])
                     for r in range(5) if f'w_cc_rand{r}' in z.files])
    ax.fill_between(dts, rand.min(0), rand.max(0), color=C_RAND, alpha=0.22,
                    label='随机图零基准区间（5 次，度数分布保留）', zorder=1)
    for k, nm, c, lw, ls in [
            ('w_eq', '等权基准', C_EQ, 1.8, '-'),
            ('w_sam', '谱切分：样本协方差', '#B0B0B0', 1.4, '--'),
            ('w_mgl', '谱切分：联合估计协方差', C_MGL, 2.0, '-'),
            ('w_cc', '联合估计 + 簇内度数倒数', C_RAND, 2.0, '-'),
            ('w_mv', '最小方差（多头）', C_MV, 2.4, '-')]:
        nav = np.cumprod(1 + z[k])
        ann = nav[-1] ** (12 / len(nav)) - 1
        sh = z[k].mean() / (z[k].std(ddof=1) + 1e-12) * np.sqrt(12)
        ax.plot(dts, nav, color=c, lw=lw, ls=ls, zorder=3,
                label=f'{nm}: {ann:+.2%} / 夏普 {sh:.2f}')
    ax.axhline(1.0, color='#888', lw=0.8)
    ax.set_ylabel('累计净值')
    ax.set_title('A 股 300 只流动性最好的股票，月频，双边 20bp（2018-01 ~ 2026-08）',
                 fontsize=12.5)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
    ax.grid(alpha=0.25, ls='--')
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.2f}'))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    cd = f'{CHART_DIR}/systematic_half'
    os.makedirs(cd, exist_ok=True)
    chart_sparsity(f'{cd}/industry_vs_sparsity.png')
    chart_nav(f'{cd}/allocation_nav.png')
    print('已写 charts/systematic_half/{industry_vs_sparsity,allocation_nav}.png')
