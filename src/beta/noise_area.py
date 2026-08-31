"""★复现: 公开研究)「另类ETF交易策略: 日内动量——指数化配置系列研究」噪声区间突破。

构造(按 公开研究原式还原, 公开研究未逐字写明的部分在 README 已声明):
  σ(T)      = 过去14个交易日 |close_d(T)/open_d − 1| 的均值   (严格排除当日, 无前视)
  Upper(T)  = open_today × (1+σ(T));  Lower(T) = open_today × (1−σ(T))
  决策时点   = 10:29 / 11:29 / 13:59 的 bar 收盘; ★成交在下一根 bar 的开盘(决策与成交分离)
  入场       = 空仓时 close(T) > Upper → 做多; < Lower → 做空
  离场       = 逐分钟检查触及反向边界 → 该分钟下一根开盘平; 否则 14:57 收盘平。不留隔夜。
  改进版v1   = 止损改为 多头 max(Lower, VWAP) / 空头 min(Upper, VWAP), VWAP 为当日累计成交额/成交量
  费用       = 每边 fee_bp; 进出各收一次

★测试段 2024-08-01 起【封存】: 本脚本 END 默认 2024-07-31, 不计算测试段。
分段: 复现对拍段 = 全窗(2014-01~2024-07-31, 最接近公开研究 2013-01-25~2024-07-31);
      项目纪律段 = 训 2016-01-01~2022-02-14 / 验 2022-02-15~2024-07-31 (剔除2015, 项目规约)
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PANEL = os.path.join(PROJ, 'data', 'idx1m.parquet')
END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp('2024-07-31')
TEST0 = pd.Timestamp('2024-08-01')          # ★封存起点
TR0, TRm = pd.Timestamp('2016-01-01'), pd.Timestamp('2022-02-15')
DEC = [1029, 1129, 1359]                    # 决策时点
CLOSE_HM = 1457                             # 平仓时刻(避开收盘集合竞价)
NDAY = 14
DPY = 244
lines = []
def P(s=''):
    print(s); lines.append(s)

assert END < TEST0, f'END={END.date()} 越过封存起点 {TEST0.date()}'

df = pd.read_parquet(PANEL)
df['dt'] = pd.to_datetime(df['date'])
df = df[df['dt'] <= END]

def run_one(g, fee_bp, improved):
    """单指数逐日执行, 返回 (逐日收益 Series, 单笔列表)。"""
    fee = fee_bp * 1e-4
    dates = g['date'].unique()
    piv_c = g.pivot_table(index='date', columns='hm', values='close')
    piv_o = g.pivot_table(index='date', columns='hm', values='open')
    piv_m = g.pivot_table(index='date', columns='hm', values='money')
    piv_v = g.pivot_table(index='date', columns='hm', values='volume')
    hms = np.array(sorted(piv_c.columns))
    first_hm = hms[0]
    day_open = piv_o[first_hm]
    # σ(T): 过去14日同时刻 |close(T)/open − 1| 均值, shift(1) 排除当日
    sig = {}
    for T in DEC:
        if T not in piv_c.columns: continue
        mv = (piv_c[T] / day_open - 1).abs()
        sig[T] = mv.rolling(NDAY, min_periods=NDAY).mean().shift(1)
    # VWAP 累计
    cm = piv_m[hms].cumsum(axis=1); cv = piv_v[hms].cumsum(axis=1)
    vwap = (cm / cv.replace(0, np.nan))
    idx_after = {T: hms[hms > T][0] if (hms > T).any() else None for T in set(DEC) | set(hms.tolist())}
    rets, trades = [], []
    C = piv_c.values; O = piv_o.values
    hm_pos = {h: i for i, h in enumerate(hms)}
    for di, d in enumerate(piv_c.index):
        o0 = day_open.iloc[di]
        if not np.isfinite(o0): rets.append(0.0); continue
        row_c, row_o = C[di], O[di]
        vw = vwap.iloc[di].values
        day_ret, pos, entry_px, entry_i, side = 0.0, 0, np.nan, -1, 0
        bnd_up = bnd_dn = np.nan
        for T in DEC:
            if T not in sig or not np.isfinite(sig[T].iloc[di]): continue
            if T not in hm_pos: continue
            iT = hm_pos[T]
            s = sig[T].iloc[di]
            up, dn = o0 * (1 + s), o0 * (1 - s)
            cT = row_c[iT]
            if not np.isfinite(cT): continue
            if pos != 0: continue                     # 已持仓则跳过该决策点
            nxt = iT + 1
            if nxt >= len(hms) or hms[nxt] > CLOSE_HM: continue
            if cT > up:   side = 1
            elif cT < dn: side = -1
            else: continue
            entry_px = row_o[nxt]
            if not np.isfinite(entry_px): side = 0; continue
            pos, entry_i, bnd_up, bnd_dn = side, nxt, up, dn
            # 持仓: 逐分钟检查止损/边界, 否则持到 CLOSE_HM
            exit_px, exit_i = np.nan, None
            for j in range(nxt, len(hms)):
                if hms[j] > CLOSE_HM: break
                cj = row_c[j]
                if not np.isfinite(cj): continue
                if improved:
                    stop = max(bnd_dn, vw[j]) if pos > 0 else min(bnd_up, vw[j])
                else:
                    stop = bnd_dn if pos > 0 else bnd_up
                hit = (cj < stop) if pos > 0 else (cj > stop)
                if hit and j + 1 < len(hms) and hms[j + 1] <= CLOSE_HM:
                    exit_px, exit_i = row_o[j + 1], j + 1; break
            if exit_i is None:
                jc = np.where(hms <= CLOSE_HM)[0][-1]
                exit_px, exit_i = row_c[jc], jc
            if np.isfinite(exit_px):
                r = pos * (exit_px / entry_px - 1) - 2 * fee
                day_ret += r
                trades.append(r)
            pos = 0
        rets.append(day_ret)
    return pd.Series(rets, index=pd.to_datetime(piv_c.index)), np.array(trades)

def metrics(r, tr, lab):
    n = len(r)
    ann = r.mean() * DPY * 100
    vol = r.std() * np.sqrt(DPY) * 100
    sh = ann / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod(); mdd = ((eq / eq.cummax()) - 1).min() * 100
    cal = ann / abs(mdd) if mdd < 0 else np.nan
    inmkt = (r != 0).mean() * 100
    if len(tr):
        w = tr > 0; aw = tr[w].mean() if w.any() else 0; al = tr[~w].mean() if (~w).any() else 0
        pf = tr[w].sum() / abs(tr[~w].sum()) if (w.any() and (~w).any()) else np.inf
        ts = (f'{len(tr):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{tr.mean()*1e4:>+6.1f}bp '
              f'盈亏比{abs(aw/al) if al else np.inf:>5.2f} PF{pf:>5.2f}')
    else:
        ts = '无交易'
    return f'{lab:<22} 年化{ann:>+6.2f}% 波动{vol:>5.2f}% Sharpe{sh:>6.2f} MDD{mdd:>7.2f}% Calmar{cal:>6.2f} 在场{inmkt:>4.0f}% | {ts}'

CODES = ['SSE50', 'CSI300', 'CSI500', 'CSI1000', 'CNI2000']
RES = {}
for fee_bp, feelab in [(1.0, '单边1bp(公开研究现货口径)'), (1.5, '单边1.5bp(公开研究期货口径)'), (10.0, '单边10bp=双边20bp(本项目标准)')]:
    for improved in [False, True]:
        vlab = '改进版(VWAP止损)' if improved else '基础版'
        P('=' * 132)
        P(f'★ {feelab} · {vlab}   [复现对拍段 {df["date"].min()} ~ {END.date()}]  ★测试段 2024-08 起已封存未计算')
        for c in CODES:
            g = df[df['code'] == c]
            if not len(g): continue
            r, tr = run_one(g, fee_bp, improved)
            P('  ' + metrics(r, tr, c))
            RES[(fee_bp, improved, c)] = (r, tr)
P()
P('=' * 132)
P('★ 项目纪律分段(剔除2015; 训 2016-01-01~2022-02-14 / 验 2022-02-15~2024-07-31) @ 双边20bp')
for improved in [False, True]:
    P(f'  ── {"改进版" if improved else "基础版"}')
    for c in CODES:
        if (10.0, improved, c) not in RES: continue
        r, _ = RES[(10.0, improved, c)]
        out = []
        for lab, m in [('训', (r.index >= TR0) & (r.index < TRm)), ('验', (r.index >= TRm) & (r.index <= END))]:
            rr = r[m]
            ann = rr.mean() * DPY * 100; vol = rr.std() * np.sqrt(DPY) * 100
            out.append(f'{lab} 年化{ann:>+6.2f}% Sharpe{(ann/vol if vol>0 else np.nan):>6.2f}')
        P(f'    {c:<10}' + ' | '.join(out))
os.makedirs(os.path.join(PROJ, 'output'), exist_ok=True)
open(os.path.join(PROJ, 'output', 'replication.txt'), 'w').write('\n'.join(lines))
import pickle
pickle.dump({k: (v[0], v[1]) for k, v in RES.items()}, open(os.path.join(PROJ, 'output', 'series.pkl'), 'wb'))
print('\nsaved output/replication.txt')
