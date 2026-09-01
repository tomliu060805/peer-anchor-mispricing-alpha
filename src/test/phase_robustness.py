# -*- coding: utf-8 -*-
"""调仓相位稳健性: 5 日调仓网格换一个起点, 业绩变不变。

动机: 回测的信号网格是 `arange(WLIM+1, T-6, 5)`, 起点 t=251 纯属任意。
实盘从哪天开始, 就把相位钉在哪天。如果业绩对相位敏感, 说明回测那条曲线
有相当部分来自"恰好在这些日子调仓", 而这在实盘不可复制。

判据: 5 个相位的超额年化极差 / 均值 < 30% 视为不敏感; 且没有相位为负。
本检验只读已有配置, 不调任何参数。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ

import os, json
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np

_src = open(f'{PROJ}/src/test/open_test_segment.py', encoding='utf-8').read()
exec(_src.split("out = {'criteria': CRIT}")[0])

SEG = {'dev': ('2016-01-01', '2022-12-31'),
       'val': ('2023-01-01', '2024-08-16'),
       'test': ('2024-08-19', '2026-12-31')}

# 锚变量只在原信号网格的日期上算过, 换相位就是另一批日期——不补算的话
# PG_b/ZS_b/PG_dual/PG_te 全是 NaN, 每期 hard.sum()<100 直接跳过, recs 为空。
def backfill(ts):
    """为给定日期补算四个锚变量, 公式与 open_test_segment 完全一致。"""
    todo = [t for t in ts if not np.isfinite(PG_b[t]).any()]
    for _t in todo:
        _g, _pm = _gap_from(RB8, NB8, WV8, _t)
        PG_b[_t] = _g
        PM_b[_t] = _pm
        PG_dual[_t] = _gap_from(_RBd, _NBd, _WVd, _t)[0]
        PG_te[_t] = _gap_from(_RBt, _NBt, _WVt, _t)[0]
    _W = 120
    for _b in range(len(RB8)):
        _t1 = int(RB8[_b]); _t0 = _t1 - _W
        _t2 = int(RB8[_b + 1]) if _b + 1 < len(RB8) else T
        _sds = [t for t in todo if _t1 <= t < _t2 and t < T]
        if not _sds:
            continue
        _nbr = NB8[_b]; _idx = np.where(_nbr[:, 0] >= 0)[0]
        if len(_idx) < 100:
            continue
        _nb = _nbr[_idx]
        _base = logc[_t0 - 1] if _t0 > 0 else np.zeros(N)
        _Pw = np.exp(logc[_t0:_t1] - _base)
        _nbc = np.where(_nb >= 0, _nb, 0)
        _sj = _Pw[:, _nbc.ravel()].reshape(_W, len(_idx), _nb.shape[1])
        _str = _Pw[:, _idx][:, :, None] - _sj
        _mu = _str.mean(0); _sd = _str.std(0) + 1e-9
        _d = (_str ** 2).sum(0); _d = _d - _d.min(1, keepdims=True)
        _wsm = np.exp(-_d) * (_nb >= 0)
        _wsm = _wsm / np.maximum(_wsm.sum(1, keepdims=True), 1e-12)
        for _t in _sds:
            _Pt = np.exp(logc[_t] - _base)
            ZS_b[_t, _idx] = -np.nansum(((_Pt[_idx][:, None] - _Pt[_nbc] - _mu) / _sd) * _wsm, 1)
    return len(todo)


res = {}
for ph in range(5):
    sig_all = [int(x) for x in np.arange(int(RB8[0]) + 1 + ph, T - 6, 5)]
    n_new = backfill(sig_all)
    print(f'相位{ph}: 补算 {n_new} 期锚变量', flush=True)
    recs = run_alpha()
    if not recs:
        print(f'相位{ph}: 无有效期, 跳过', flush=True)
        continue
    ds = np.array([r[0] for r in recs])
    pr = np.array([r[1] for r in recs])
    bq = np.array([r[2] for r in recs])
    d = {}
    for k, (a, b) in SEG.items():
        m = (ds >= np.datetime64(a)) & (ds <= np.datetime64(b))
        if m.sum() < 10:
            continue
        ex = (1 + pr[m]) / (1 + bq[m]) - 1
        nav = np.cumprod(1 + ex)
        d[k] = {'ExAnn': round(float(nav[-1] ** (50.4 / len(ex)) - 1), 4),
                'IR': round(float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(50.4)), 2),
                'n': int(m.sum())}
    res[f'phase{ph}'] = d
    print(f'相位{ph} (起点偏移{ph}日, {len(sig_all)}期): ' +
          '  '.join(f'{k} 超额{v["ExAnn"]:+.1%} IR{v["IR"]:.2f}' for k, v in d.items()), flush=True)

print()
verdict = {}
for k in SEG:
    v = [res[f'phase{p}'][k]['ExAnn'] for p in range(5) if k in res[f'phase{p}']]
    i = [res[f'phase{p}'][k]['IR'] for p in range(5) if k in res[f'phase{p}']]
    if not v:
        continue
    spread = (max(v) - min(v)) / abs(np.mean(v)) if np.mean(v) != 0 else np.inf
    verdict[k] = {'ExAnn_mean': round(float(np.mean(v)), 4),
                  'ExAnn_min': round(min(v), 4), 'ExAnn_max': round(max(v), 4),
                  'rel_spread': round(float(spread), 3),
                  'IR_mean': round(float(np.mean(i)), 2),
                  'IR_min': round(min(i), 2), 'IR_max': round(max(i), 2),
                  'any_negative': bool(min(v) <= 0),
                  'pass': bool(spread < 0.30 and min(v) > 0)}
    print(f'{k}: 超额 {min(v):+.1%} ~ {max(v):+.1%} (均值 {np.mean(v):+.1%}, '
          f'相对极差 {spread:.1%})  IR {min(i):.2f} ~ {max(i):.2f}  '
          f'{"不敏感" if verdict[k]["pass"] else "★敏感"}')

json.dump({'by_phase': res, 'verdict': verdict},
          open(f'{PROJ}/output/alpha/phase_robustness.json', 'w'), ensure_ascii=False, indent=1)
print('\n已写 output/alpha/phase_robustness.json')
