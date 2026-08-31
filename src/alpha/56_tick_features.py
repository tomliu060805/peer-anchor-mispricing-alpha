# -*- coding: utf-8 -*-
"""v30a: 逐笔行为特征构建(全市场逐日). 每股每日:
- 主动卖/买 按委托号聚合: 扫单金额(单一主动单吃掉>=3个对手单), 主动单数, 主动金额
输出 cache/tickbehav/{date}.parquet"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, TICK_ROOT
import os,sys,datetime as dt
os.environ['POLARS_MAX_THREADS']='2'
import numpy as np, polars as pl
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache/tickbehav'
RAW=TICK_ROOT+''
g=np.load(PROJ+'/cache/daily_grid.npz')
dates=[str(d) for d in g['dates'] if '2015-12-01'<=str(d)<='2026-08-31']
PREF=("00","30","60","68")
def one_side(tr,idcol,side_label):
    per=(tr.group_by(['code',idcol]).agg(pl.col('amount').sum().alias('amt'),pl.len().alias('pieces')))
    return (per.group_by('code').agg(
        pl.col('amt').sum().alias(f'{side_label}_amt'),
        pl.col('amt').filter(pl.col('pieces')>=3).sum().alias(f'{side_label}_sweep'),
        pl.len().alias(f'{side_label}_orders')))
def build(ds):
    out=f'{CACHE}/{ds}.parquet'
    if os.path.exists(out): return ds,'skip'
    try:
        parts=[]
        shp=f'{RAW}/SEL2_TRANSACTION/{ds}.parquet'
        if os.path.exists(shp):
            sh=(pl.scan_parquet(shp)
                .select(pl.col('Symbol').alias('code'),pl.col('TradeAmount').fill_null(0.0).alias('amount'),
                        pl.col('BuySellFlag').alias('flag'),pl.col('BuyRecID'),pl.col('SellRecID'))
                .filter(pl.col('code').str.slice(0,2).is_in(list(PREF))))
            tm=sh.group_by('code').agg(pl.col('amount').sum().alias('total_money'))
            s_ag=one_side(sh.filter(pl.col('flag')=='S').rename({'SellRecID':'oid'}),'oid','asell')
            b_ag=one_side(sh.filter(pl.col('flag')=='B').rename({'BuyRecID':'oid'}),'oid','abuy')
            d=tm.join(s_ag,on='code',how='left').join(b_ag,on='code',how='left').with_columns(pl.lit('SH').alias('exchange'))
            parts.append(d.collect(engine='streaming'))
        szp=f'{RAW}/SZL2_TRADE/{ds}.parquet'
        if os.path.exists(szp):
            sz=(pl.scan_parquet(szp)
                .filter(~(pl.col('TradeType')=='4').fill_null(False))
                .select(pl.col('Symbol').alias('code'),
                        (pl.col('TradePrice')*pl.col('TradeVolume')).fill_null(0.0).alias('amount'),
                        pl.col('BuyOrderID'),pl.col('SellOrderID'))
                .filter(pl.col('code').str.slice(0,2).is_in(list(PREF))))
            tm=sz.group_by('code').agg(pl.col('amount').sum().alias('total_money'))
            s_ag=one_side(sz.filter(pl.col('SellOrderID')>pl.col('BuyOrderID')).rename({'SellOrderID':'oid'}),'oid','asell')
            b_ag=one_side(sz.filter(pl.col('BuyOrderID')>pl.col('SellOrderID')).rename({'BuyOrderID':'oid'}),'oid','abuy')
            d=tm.join(s_ag,on='code',how='left').join(b_ag,on='code',how='left').with_columns(pl.lit('SZ').alias('exchange'))
            parts.append(d.collect(engine='streaming'))
        if not parts: return ds,'nofile'
        allp=pl.concat(parts,how='vertical').with_columns(pl.lit(ds).alias('date'))
        allp.write_parquet(out+'.tmp'); os.rename(out+'.tmp',out)
        return ds,'ok'
    except Exception as e:
        return ds,f'ERR:{type(e).__name__}:{e}'
if __name__=='__main__':
    with ProcessPoolExecutor(50) as ex:
        for i,(ds,st) in enumerate(ex.map(build,dates,chunksize=1)):
            if st.startswith('ERR') or i%100==0: print(i,ds,st,flush=True)
    print('tick features done')
