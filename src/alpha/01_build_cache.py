# -*- coding: utf-8 -*-
"""构建日频宽表缓存: 收益率/停牌/涨跌停/ST/市值/行业.
收益 = close/pre_close - 1 (pre_close 为复权口径的昨收, 自动处理除权除息)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, STOCK_ROOT
import os, sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = STOCK_ROOT+''
OUT = PROJ+'/cache'
os.makedirs(OUT, exist_ok=True)

def load_day(f):
    df = pd.read_parquet(f, columns=['date','code','close','pre_close','paused','high_limit','low_limit','high','low','open','money'])
    return df

def main():
    files = sorted(os.listdir(f'{ROOT}/price/price_daily'))
    files = [f'{ROOT}/price/price_daily/{f}' for f in files if f.endswith('.parquet')]
    print(f'{len(files)} daily files')
    with ProcessPoolExecutor(12) as ex:
        parts = list(ex.map(load_day, files, chunksize=32))
    df = pd.concat(parts, ignore_index=True)
    print('long shape', df.shape)
    df['date'] = pd.to_datetime(df['date'])
    df['ret'] = df['close']/df['pre_close'] - 1.0
    # 板别: 涨停触及(收盘=涨停价)
    df['at_hlimit'] = (df['close'] >= df['high_limit'] - 1e-9)
    df['at_llimit'] = (df['close'] <= df['low_limit'] + 1e-9)

    piv = lambda col, dt: df.pivot_table(index='date', columns='code', values=col, aggfunc='first').astype(dt)
    ret = piv('ret', 'float32')
    close = piv('close', 'float32')
    opn = piv('open', 'float32')
    money = piv('money', 'float32')
    paused = piv('paused', 'float32')
    hl = piv('at_hlimit', 'float32')
    ll = piv('at_llimit', 'float32')

    dates = ret.index.strftime('%Y-%m-%d').values.astype('U10')
    codes = ret.columns.values.astype('U11')
    np.savez_compressed(f'{OUT}/daily_grid.npz',
        dates=dates, codes=codes,
        ret=ret.values, close=close.values, open=opn.values, money=money.values,
        paused=paused.values, at_hlimit=hl.values, at_llimit=ll.values)
    print('saved daily_grid.npz', ret.shape)

    # ST 宽表 (逐日文件)
    stf = sorted(os.listdir(f'{ROOT}/info/st_info'))
    with ProcessPoolExecutor(12) as ex:
        sts = list(ex.map(pd.read_parquet, [f'{ROOT}/info/st_info/{f}' for f in stf], chunksize=64))
    st = pd.concat(sts, ignore_index=True)
    st['date'] = pd.to_datetime(st['date'])
    stp = st.pivot_table(index='date', columns='code', values='is_st', aggfunc='first').astype('float32')
    stp = stp.reindex(index=ret.index, columns=ret.columns)
    np.savez_compressed(f'{OUT}/st_grid.npz', dates=dates, codes=codes, is_st=stp.values)
    print('saved st_grid.npz')

    # 市值 (月末刷新即可, 但直接存日频 float32 也不大)
    vf = sorted(os.listdir(f'{ROOT}/fundamental/valuation'))
    def lv(f): return pd.read_parquet(f, columns=['date','code','circulating_market_cap','turnover_ratio'])
    with ProcessPoolExecutor(12) as ex:
        vs = list(ex.map(lv, [f'{ROOT}/fundamental/valuation/{f}' for f in vf], chunksize=64))
    val = pd.concat(vs, ignore_index=True)
    val['date'] = pd.to_datetime(val['date'])
    mc = val.pivot_table(index='date', columns='code', values='circulating_market_cap', aggfunc='first').astype('float32').reindex(index=ret.index, columns=ret.columns)
    np.savez_compressed(f'{OUT}/mcap_grid.npz', dates=dates, codes=codes, mcap=mc.values)
    print('saved mcap_grid.npz')

    # 行业 (jq_l1, 月度采样: 每月第一个交易日)
    inf = sorted(os.listdir(f'{ROOT}/info/industry'))
    ind_map = {}
    month_first = ret.index.to_series().groupby(ret.index.to_period('M')).first()
    keep = set(d.strftime('%Y-%m-%d')+'.parquet' for d in month_first)
    use = [f for f in inf if f in keep]
    for f in use:
        d = pd.read_parquet(f'{ROOT}/info/industry/{f}')
        d = d[d['category']=='sw_l1']
        ind_map[f[:10]] = dict(zip(d['code'], d['industry_name']))
    if not ind_map:  # fallback jq_l1
        for f in use:
            d = pd.read_parquet(f'{ROOT}/info/industry/{f}')
            d = d[d['category']=='jq_l1']
            ind_map[f[:10]] = dict(zip(d['code'], d['industry_name']))
    import pickle
    with open(f'{OUT}/industry_monthly.pkl','wb') as fh:
        pickle.dump(ind_map, fh)
    print('saved industry_monthly.pkl months=', len(ind_map))

if __name__ == '__main__':
    main()
