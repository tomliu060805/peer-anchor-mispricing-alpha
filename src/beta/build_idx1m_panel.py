"""指数 1m 面板构建(复现公开研究)。
标的: 上证50/沪深300/中证500/中证1000/国证2000。源: ${INDEX_ROOT}/price/price_1m(官方)。
产出: data/idx1m.parquet, 列 = date,code,hm,open,high,low,close,money,volume
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import INDEX_ROOT
import os, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

SRC = INDEX_ROOT+'/price/price_1m'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'idx1m.parquet')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CODES = {'000016.XSHG': 'SSE50', '000300.XSHG': 'CSI300', '000905.XSHG': 'CSI500',
         '000852.XSHG': 'CSI1000', '399303.XSHE': 'CNI2000'}

def one(f):
    try:
        d = pd.read_parquet(f, columns=['datetime', 'code', 'open', 'high', 'low', 'close', 'money', 'volume'])
    except Exception:
        return None
    d = d[d['code'].astype(str).isin(CODES)]
    if not len(d):
        return None
    dt = pd.to_datetime(d['datetime'])
    d = d.assign(date=dt.dt.strftime('%Y-%m-%d'),
                 hm=(dt.dt.hour * 100 + dt.dt.minute).astype('int16'))
    return d.drop(columns=['datetime'])

fs = sorted(glob.glob(f'{SRC}/*.parquet'))
print(f'扫描 {len(fs)} 天 {os.path.basename(fs[0])[:10]} ~ {os.path.basename(fs[-1])[:10]}', flush=True)
rows = []
with ProcessPoolExecutor(max_workers=int(os.environ.get('NW', '16'))) as ex:
    for i, r in enumerate(ex.map(one, fs, chunksize=8)):
        if r is not None:
            rows.append(r)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(fs)}', flush=True)
df = pd.concat(rows, ignore_index=True)
df['code'] = df['code'].astype(str).map(CODES)
df = df.sort_values(['code', 'date', 'hm']).reset_index(drop=True)
print(df.groupby('code')['date'].agg(['min', 'max', 'nunique']).to_string())
df.to_parquet(OUT, index=False)
print(f'落盘 {os.path.abspath(OUT)} {df.shape}')
