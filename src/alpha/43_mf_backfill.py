# -*- coding: utf-8 -*-
"""v24a: 补算2016-2018日频资金流(复用主题库构建函数, 写本项目cache不动其库)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, EXT_THEME_ROOT as EXT_THEME
import os,sys,datetime as dt
os.environ['POLARS_MAX_THREADS']='3'
import numpy as np
sys.path.insert(0, EXT_THEME)
from src.data.moneyflow_l2 import compute_moneyflow_daily
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache/moneyflow_ext'
g=np.load(PROJ+'/cache/daily_grid.npz')
dates=[str(d) for d in g['dates'] if '2016-01-01'<=str(d)<='2018-12-31']
def build(ds):
    out=f'{CACHE}/{ds}.parquet'
    if os.path.exists(out): return ds,'skip'
    try:
        df=compute_moneyflow_daily(ds)
        df.write_parquet(out+'.tmp'); os.rename(out+'.tmp',out)
        return ds,'ok'
    except Exception as e:
        return ds,f'ERR:{e}'
if __name__=='__main__':
    with ProcessPoolExecutor(30) as ex:
        for i,(ds,st) in enumerate(ex.map(build,dates,chunksize=1)):
            if st.startswith('ERR') or i%50==0: print(i,ds,st,flush=True)
    print('backfill done')
