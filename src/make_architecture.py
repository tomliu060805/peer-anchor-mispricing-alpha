# -*- coding: utf-8 -*-
"""项目架构图: 四阶段全流程一页纸。

产出 charts/architecture/strategy_architecture.png
数字口径全部取自 config/frozen_alpha_v28.json, 改配置后重跑本脚本即可同步。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import PROJ_ROOT as PROJ, CHART_DIR

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色: 三块打分各一色, 与后续图表一致
C_HEAD = '#2F5573'      # 阶段标题条
C_ANCH = '#4C72B0'      # 五锚 40%
C_BEHA = '#DD8452'      # 微观行为 40%
C_STRU = '#55A868'      # 结构 20%
C_BETA = '#8C8C8C'      # beta 择时层
C_PANEL = '#EDF1F5'     # 阶段底板
C_BOX = '#FFFFFF'
C_LINE = '#B8C4D0'
C_NOTE = '#F2F2F2'

W, H = 100.0, 58.0


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_LINE, tc='#1A1A1A',
        fs=8.2, lw=1.0, weight='normal', pad=0.35, va='center', align='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f'round,pad={pad},rounding_size=0.6',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ha = {'center': 'center', 'left': 'left'}[align]
    tx = x + w / 2 if align == 'center' else x + 0.9
    ty = y + h / 2 if va == 'center' else y + h - 1.0
    ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs, color=tc,
            weight=weight, zorder=3, linespacing=1.5)


def header(ax, x, y, w, h, text, fs=12.5):
    box(ax, x, y, w, h, text, fc=C_HEAD, ec=C_HEAD, tc='white', fs=fs, weight='bold')


def panel(ax, x, y, w, h, fc=C_PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=0.8',
                                fc=fc, ec='#D5DDE5', lw=1.0, zorder=1))


def arrow(ax, p0, p1, style='-|>', color='#5A6B7C', lw=1.6, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=4,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=2, shrinkB=2))


def build():
    fig, ax = plt.subplots(figsize=(20, 11.6))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    fig.patch.set_facecolor('white')

    # ================= 标题 + 图例 =================
    box(ax, 14, 50.6, 46, 6.2,
        '联动锚定错价选股 + 破位择时\n'
        '在“活跃打板域”内，用同伴网络定位错价、用逐笔行为筛掉知情抛售',
        fc='#F7F9FB', ec='#C9D4DE', fs=15, weight='bold')

    lx, ly = 63.5, 50.4
    box(ax, lx, ly, 34.5, 6.6, '', fc='#FAFBFC', ec='#D5DDE5')
    for i, (c, t) in enumerate([
            (C_ANCH, '五锚 40%：同伴网络定位“该涨没涨”'),
            (C_BEHA, '微观行为 40%：逐笔剔除知情抛售'),
            (C_STRU, '结构 20%：小盘误差过滤'),
            (C_BETA, 'beta 层：指数破位空头（融合 h=1.0）')]):
        yy = ly + 5.5 - i * 1.42
        ax.add_patch(FancyBboxPatch((lx + 1.0, yy - 0.42), 1.5, 0.84,
                                    boxstyle='round,pad=0.05,rounding_size=0.2',
                                    fc=c, ec=c, zorder=3))
        ax.text(lx + 3.1, yy, t, fontsize=9.3, va='center', color='#1A1A1A', zorder=3)

    # ================= 阶段一 =================
    X1, W1 = 1.2, 14.5
    header(ax, X1, 45.0, W1, 3.4, '阶段一：域的界定')
    panel(ax, X1, 12.0, W1, 31.8)
    box(ax, X1 + 1.0, 38.6, W1 - 2.0, 4.2,
        '数据层（全部 PIT）\n日线 · L2逐笔 · 财务 · 30分钟 · 指数',
        fs=8.6, weight='bold')
    box(ax, X1 + 1.0, 31.6, W1 - 2.0, 6.0,
        '1. 活跃域\n过去 250 日\n收盘封涨停 ≥ 2 次\n（约 2500 只）', fs=8.8)
    box(ax, X1 + 1.0, 24.4, W1 - 2.0, 6.2,
        '2. 硬剔（仅机制性不可买）\n停牌 / 当日封涨停\nST / 域外', fs=8.8)
    box(ax, X1 + 1.0, 17.6, W1 - 2.0, 5.8,
        '3. 其余判断一律软化\n为分数而非过滤\n（硬过滤浪费信息）', fs=8.8)
    box(ax, X1 + 1.0, 12.8, W1 - 2.0, 3.8, '当期候选池', fc=C_HEAD, ec=C_HEAD,
        tc='white', fs=9.5, weight='bold')

    # ================= 阶段二 =================
    X2, W2 = 17.6, 36.0
    header(ax, X2, 45.0, W2, 3.4, '阶段二：三块打分（全部转秩后加权）')
    panel(ax, X2, 5.4, W2, 38.4)

    box(ax, X2 + 1.2, 38.2, W2 - 2.4, 4.8,
        '联动网络：120 日窗口，剔市场 beta 与申万一级行业均值后取残差相关\n'
        'top-K=5 邻居，每 21 个交易日重建 → 价格 / 价量双确认 / 转移熵 三张网',
        fs=8.7, weight='bold', fc='#E8EEF5', ec='#BCCBDA')

    # ---- 五锚 ----
    ax.add_patch(FancyBboxPatch((X2 + 1.2, 12.2), 17.0, 24.6,
                                boxstyle='round,pad=0.3,rounding_size=0.7',
                                fc='#FFFFFF', ec=C_ANCH, lw=1.8, zorder=2))
    box(ax, X2 + 2.0, 33.0, 15.4, 2.9, '五锚  权重 40%', fc=C_ANCH, ec=C_ANCH,
        tc='white', fs=10, weight='bold')
    for i, t in enumerate([
            '价格 gap ÷ 波动\n邻居动量均值 − 自身',
            'zspread\n配对价差 z 的距离加权聚合',
            '价量双确认 gap\n价格与净流入同时强连边',
            '转移熵有向 gap\n状态离散化后的方向性传导',
            'ROE gap\n自身 − 邻居（方向与价格锚相反）']):
        box(ax, X2 + 2.0, 29.4 - i * 3.6, 15.4, 3.1, t, fs=7.9,
            fc='#F4F7FB', ec='#C6D4E4')

    # ---- 微观行为 ----
    ax.add_patch(FancyBboxPatch((X2 + 19.0, 24.0), 15.8, 12.8,
                                boxstyle='round,pad=0.3,rounding_size=0.7',
                                fc='#FFFFFF', ec=C_BEHA, lw=1.8, zorder=2))
    box(ax, X2 + 19.8, 33.0, 14.2, 2.9, '微观行为  权重 40%', fc=C_BEHA, ec=C_BEHA,
        tc='white', fs=10, weight='bold')
    box(ax, X2 + 19.8, 29.2, 14.2, 3.4,
        '− 扫单卖占比（前 5 日）\n单笔主动单吃掉 ≥3 张对手单的金额占比',
        fs=7.9, fc='#FDF3EC', ec='#E8C6AC')
    box(ax, X2 + 19.8, 25.0, 14.2, 3.4,
        '− 卖单平均规模（前 5 日）\n主动卖金额 ÷ 主动卖母单数',
        fs=7.9, fc='#FDF3EC', ec='#E8C6AC')

    # ---- 结构 ----
    ax.add_patch(FancyBboxPatch((X2 + 19.0, 12.2), 15.8, 10.6,
                                boxstyle='round,pad=0.3,rounding_size=0.7',
                                fc='#FFFFFF', ec=C_STRU, lw=1.8, zorder=2))
    box(ax, X2 + 19.8, 19.0, 14.2, 2.9, '结构  权重 20%', fc=C_STRU, ec=C_STRU,
        tc='white', fs=10, weight='bold')
    box(ax, X2 + 19.8, 15.8, 14.2, 2.7, '股价 ≥ 2 元（避面值退市螺旋）',
        fs=7.9, fc='#EDF6F0', ec='#B7D8C3')
    box(ax, X2 + 19.8, 12.7, 14.2, 2.7, 'dROE 未恶化（新报告发布时）',
        fs=7.9, fc='#EDF6F0', ec='#B7D8C3')

    box(ax, X2 + 1.2, 6.2, W2 - 2.4, 4.6,
        '综合分 = 0.4 × 五锚 + 0.4 × 微观行为 + 0.2 × 结构\n'
        '（三块均先在域内转为 0–1 分位，再加权）',
        fs=10.2, weight='bold', fc=C_HEAD, ec=C_HEAD, tc='white')

    # ================= 阶段三 =================
    X3, W3 = 55.4, 19.5
    header(ax, X3, 45.0, W3, 3.4, '阶段三：组合构建')
    panel(ax, X3, 12.0, W3, 31.8)
    for i, (t, fs_) in enumerate([
            ('1. 选股\nTop N = 200', 9.0),
            ('2. 缓冲带\n排名跌出前 600 才卖\n停牌 / 跌停顺延', 8.6),
            ('3. 权重\n∝ 综合分 + 0.3', 9.0),
            ('4. 组合约束（B2）\n个股 ≤ 1%\n行业偏离 ±5%\n投影梯度 30 次迭代', 8.4)]):
        box(ax, X3 + 1.2, 37.6 - i * 6.5, W3 - 2.4, 5.4, t, fs=fs_)
    box(ax, X3 + 1.2, 12.8, W3 - 2.4, 4.0, '目标持仓  w*\n每 5 个交易日调仓',
        fc=C_HEAD, ec=C_HEAD, tc='white', fs=9.3, weight='bold')

    # ================= 阶段四 =================
    X4, W4 = 76.6, 22.2
    header(ax, X4, 45.0, W4, 3.4, '阶段四：择时与执行')
    panel(ax, X4, 12.0, W4, 31.8)
    ax.add_patch(FancyBboxPatch((X4 + 1.0, 27.8), W4 - 2.0, 15.0,
                                boxstyle='round,pad=0.3,rounding_size=0.7',
                                fc='#FFFFFF', ec=C_BETA, lw=1.8, zorder=2))
    box(ax, X4 + 1.8, 39.6, W4 - 3.6, 2.7, 'beta 层：噪声区间破位空头',
        fc=C_BETA, ec=C_BETA, tc='white', fs=9.6, weight='bold')
    box(ax, X4 + 1.8, 33.0, W4 - 3.6, 6.2,
        'σ = 过去 14 日均 |收盘/开盘 − 1|（滞后 1 日）\n'
        '噪声带 = 开盘 × (1 ± 2.5σ)\n'
        '日内 4 个时点判定，跌破下沿次 bar 开盘做空\n'
        '每信号 0.25 单位，单日至多 4 次，次日首 bar 平',
        fs=7.8, align='left')
    box(ax, X4 + 1.8, 28.4, W4 - 3.6, 3.9,
        '融合 h = 1.0：多头 1.0× + 破位空头峰值 1.0×\n改善的是超额回撤，不是绝对回撤',
        fs=8.0, fc='#F0F0F0', ec='#C8C8C8')

    box(ax, X4 + 1.0, 18.2, W4 - 2.0, 8.6,
        '执行（路径 B，默认）\n'
        '收盘后用完整数据计算\n'
        'T+1 11:30 前完成卖出腿\n'
        'T+1 13:30–14:30 完成买入腿\n'
        '禁止开盘集中下单（实测贵 13–17bp/笔）',
        fs=8.4)
    box(ax, X4 + 1.0, 12.8, W4 - 2.0, 4.4,
        '备用路径 A：14:50 实时算（逐笔用滞后窗）+ 尾盘执行',
        fs=8.2, fc=C_NOTE, ec='#D8D8D8')

    # ================= 底部：验证纪律 =================
    panel(ax, X2, 0.4, W - X2 - 1.2, 4.6, fc='#FAFBFC')
    ax.add_patch(FancyBboxPatch((X2 + 0.9, 1.5), 7.4, 2.4,
                                boxstyle='round,pad=0.2,rounding_size=0.4',
                                fc=C_HEAD, ec=C_HEAD, zorder=3))
    ax.text(X2 + 4.6, 2.7, '验证纪律', fontsize=10, weight='bold',
            color='white', ha='center', va='center', zorder=4)
    tx = X2 + 9.8
    ax.text(tx, 3.5,
            '分段  dev 2016–2022 / val 2023–2024.08 / test ≥2024-08-19（已一次性开封，不得再调参）'
            '　·　成本  双边 20bp　·　基准  中证全指',
            fontsize=8.5, color='#333333', va='center')
    ax.text(tx, 1.7,
            '生产实现与回测逐股对拍 5/5 日期 200/200　·　调仓相位五档全正（相对极差 ≤15.3%）'
            '　·　网络类因子一律对随机网络零基准　·　每测必先跑 PIT 审计',
            fontsize=8.5, color='#333333', va='center')

    # ================= 连线 =================
    arrow(ax, (X1 + W1, 27.5), (X2, 27.5))
    ax.text((X1 + W1 + X2) / 2, 28.6, '按域筛选', fontsize=8, ha='center', color='#5A6B7C')
    arrow(ax, (X2 + W2, 27.5), (X3, 27.5))
    ax.text((X2 + W2 + X3) / 2, 28.6, '综合分', fontsize=8, ha='center', color='#5A6B7C')
    arrow(ax, (X3 + W3, 27.5), (X4, 27.5))
    ax.text((X3 + W3 + X4) / 2, 28.6, 'w*', fontsize=8, ha='center', color='#5A6B7C')
    # 三块 -> 综合分
    for x0 in (X2 + 9.7, X2 + 26.9):
        arrow(ax, (x0, 12.2), (x0, 11.0), lw=1.3)
    # 网络 -> 五锚
    arrow(ax, (X2 + 9.7, 38.2), (X2 + 9.7, 36.9), lw=1.3)

    os.makedirs(f'{CHART_DIR}/architecture', exist_ok=True)
    out = f'{CHART_DIR}/architecture/strategy_architecture.png'
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('已写', out.replace(PROJ, '.'))
    return out


if __name__ == '__main__':
    build()
