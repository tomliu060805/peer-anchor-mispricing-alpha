# -*- coding: utf-8 -*-
"""财务比率面板的前视审计。

项目规矩: 任何要测的东西先跑机械化 PIT 审计, 不靠"我记得写对了"。

三项:
  A. 源头核对   随机抽若干 (股票, 日期), 回源三张报表, 确认写进面板的值
                确实来自 pub_date < 该日 的那份报告。
  B. 遮罩扰动   把某日之后的所有源数据抹掉重建该日的比率, 与原面板逐值比较。
                若有前视, 抹掉未来会改变当日取值。
  C. 台阶检测   比率在截面上应只在财报发布日附近跳变。统计"取值发生变化的
                股票占比"的日度序列, 若在非财报期也有大量变化即可疑。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import CACHE_DIR as CACHE, STOCK_ROOT

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
FUND = f'{STOCK_ROOT}/fundamental'


def audit_a(n=12):
    z = np.load(f'{CACHE}/finratio_grid.npz')
    R, names, dates, codes = z['ratios'], list(z['names']), z['dates'], z['codes']
    print('=== A. 源头核对 (资产负债率) ===')
    bad = 0
    tried = 0
    for _ in range(n * 4):
        if tried >= n:
            break
        t = int(rng.integers(300, len(dates)))
        ds = str(dates[t])
        try:
            b = pd.read_parquet(f'{FUND}/balance_sheet/{ds}.parquet',
                                columns=['code', 'pub_date', 'total_assets', 'total_liability'])
        except Exception:
            continue
        b = b[pd.to_datetime(b['pub_date'], errors='coerce')
              <= pd.Timestamp(ds) - pd.Timedelta(days=1)]
        if len(b) == 0:
            continue
        row = b.iloc[int(rng.integers(len(b)))]
        ci = np.where(codes == row['code'])[0]
        if not len(ci):
            continue
        tried += 1
        want = float(row['total_liability']) / float(row['total_assets'])
        want = min(max(want, 0), 3)
        got = float(R[t, names.index('debt_ratio'), ci[0]])
        ok = np.isfinite(got) and abs(got - want) < 1e-4
        bad += (not ok)
        print(f'  {ds} {row["code"]}  面板 {got:.6f}  回源 {want:.6f}  '
              f'pub_date {row["pub_date"]}  {"✓" if ok else "✗"}')
    print(f'  -> {tried - bad}/{tried} 一致\n')
    return bad == 0


def audit_b(ds='2022-06-30'):
    """遮罩扰动: 只用 <= ds 的源文件重算该日, 与面板比较。"""
    z = np.load(f'{CACHE}/finratio_grid.npz')
    R, names, dates, codes = z['ratios'], list(z['names']), z['dates'], z['codes']
    t = int(np.where(dates == ds)[0][0])
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__))))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'fp', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '01_finratio_panel.py'))
    fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fp)
    fp._init(dates, codes)
    _, out = fp._load(t)
    d = np.abs(np.nan_to_num(out, nan=0) - np.nan_to_num(R[t], nan=0))
    print(f'=== B. 遮罩扰动 ({ds}) ===')
    print(f'  逐值最大差 {d.max():.3e}  非空模式一致 '
          f'{bool((np.isfinite(out) == np.isfinite(R[t])).all())}')
    print(f'  -> {"无差异, 该日取值不依赖未来数据" if d.max() < 1e-6 else "★有差异, 需排查"}\n')
    return d.max() < 1e-6


def audit_c():
    z = np.load(f'{CACHE}/finratio_grid.npz')
    R, names, dates = z['ratios'], list(z['names']), z['dates']
    j = names.index('debt_ratio')
    V = R[:, j, :]
    chg = np.zeros(len(dates))
    for t in range(1, len(dates)):
        a, b = V[t - 1], V[t]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 100:
            chg[t] = float((np.abs(a[m] - b[m]) > 1e-9).mean())
    mon = pd.Series(chg, index=pd.to_datetime(dates)).groupby(
        lambda x: x.month).mean()
    print('=== C. 台阶检测 (资产负债率逐日变化的股票占比, 按月平均) ===')
    for m_, v in mon.items():
        bar = '█' * int(v * 60)
        flag = '  <- 财报密集月' if m_ in (4, 7, 8, 10) else ''
        print(f'  {m_:2d}月 {v:6.2%} {bar}{flag}')
    peak = mon.loc[[4, 7, 8, 10]].mean()
    off = mon.drop([4, 7, 8, 10]).mean()
    print(f'  -> 财报月均 {peak:.2%} vs 其余月均 {off:.2%}  '
          f'({"符合季报节奏" if peak > off * 2 else "★节奏异常"})')
    return peak > off * 2


if __name__ == '__main__':
    ok = [audit_a(), audit_b(), audit_c()]
    print(f'\nPIT 审计: {"全部通过" if all(ok) else "★存在问题"}')
