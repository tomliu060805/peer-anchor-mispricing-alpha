# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, STOCK_ROOT
import os, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT=STOCK_ROOT+''; OUT=PROJ+'/cache'

def lv(f): return pd.read_parquet(f, columns=['date','code','circulating_market_cap','turnover_ratio'])

def main():
    g = np.load(f'{OUT}/daily_grid.npz'); dates=g['dates']; codes=g['codes']
    didx = pd.to_datetime(dates)
    vf = sorted(os.listdir(f'{ROOT}/fundamental/valuation'))
    with ProcessPoolExecutor(12) as ex:
        vs = list(ex.map(lv, [f'{ROOT}/fundamental/valuation/{f}' for f in vf], chunksize=64))
    val = pd.concat(vs, ignore_index=True); val['date']=pd.to_datetime(val['date'])
    mc = val.pivot_table(index='date', columns='code', values='circulating_market_cap', aggfunc='first').astype('float32').reindex(index=didx, columns=codes)
    np.savez_compressed(f'{OUT}/mcap_grid.npz', dates=dates, codes=codes, mcap=mc.values)
    print('saved mcap_grid.npz')
    inf = sorted(os.listdir(f'{ROOT}/info/industry'))
    month_first = pd.Series(didx).groupby(didx.to_period('M')).first()
    keep = set(d.strftime('%Y-%m-%d')+'.parquet' for d in month_first)
    use = [f for f in inf if f in keep]
    ind_map={}
    cats_seen=set()
    for f in use:
        d = pd.read_parquet(f'{ROOT}/info/industry/{f}')
        cats_seen |= set(d['category'].unique())
        ds = d[d['category']=='sw_l1']
        if len(ds)==0: ds = d[d['category']=='jq_l1']
        ind_map[f[:10]] = dict(zip(ds['code'], ds['industry_name']))
    with open(f'{OUT}/industry_monthly.pkl','wb') as fh: pickle.dump(ind_map, fh)
    print('saved industry_monthly.pkl months=', len(ind_map), 'cats=', cats_seen)

if __name__=='__main__': main()
