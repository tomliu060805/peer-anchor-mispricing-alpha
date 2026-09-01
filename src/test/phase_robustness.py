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

_base_sig = list(sig_all)
res = {}
for ph in range(5):
    sig_all = [int(x) for x in np.arange(int(RB8[0]) + 1 + ph, T - 6, 5)]
    recs = run_alpha()
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
