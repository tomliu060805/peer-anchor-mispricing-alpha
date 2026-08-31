# -*- coding: utf-8 -*-
"""v38a: 执行轨迹特征(拆单时代口径). 按委托号聚合母单, 每股每日分主动买/卖输出:
  n_orders, amt, sweep_amt(笔数>=3), 
  lvl_amt(穿透>=3个不同价位的金额)  <- 真穿透档位
  fast_amt(母单执行时长<=1秒), slow_amt(>=60秒)
  split_amt(笔数>=5 且 单笔均额<5万 = 算法拆单特征)
  dur_wavg(金额加权执行时长), pieces_wavg
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, TICK_ROOT
import os,sys
os.environ['POLARS_MAX_THREADS']='2'
import numpy as np, polars as pl
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache/ticktraj'
RAW=TICK_ROOT+''
g=np.load(PROJ+'/cache/daily_grid.npz')
dates=[str(d) for d in g['dates'] if '2015-12-01'<=str(d)<='2026-08-31']
PREF=("00","30","60","68")
def agg_side(tr,idcol,tag):
    per=(tr.group_by(['code',idcol]).agg(
            pl.col('amount').sum().alias('amt'),
            pl.len().alias('pieces'),
            pl.col('price').n_unique().alias('lvls'),
            (pl.col('ts').max()-pl.col('ts').min()).alias('dur')))
    per=per.with_columns((pl.col('amt')/pl.col('pieces')).alias('avg_piece'))
    return per.group_by('code').agg(
        pl.len().cast(pl.Int64).alias(f'{tag}_n'),
        pl.col('amt').sum().alias(f'{tag}_amt'),
        pl.col('amt').filter(pl.col('pieces')>=3).sum().alias(f'{tag}_sweep'),
        pl.col('amt').filter(pl.col('lvls')>=3).sum().alias(f'{tag}_lvl3'),
        pl.col('amt').filter(pl.col('dur')<=1.0).sum().alias(f'{tag}_fast'),
        pl.col('amt').filter(pl.col('dur')>=60.0).sum().alias(f'{tag}_slow'),
        pl.col('amt').filter((pl.col('pieces')>=5)&(pl.col('avg_piece')<50000)).sum().alias(f'{tag}_split'),
        ((pl.col('amt')*pl.col('dur')).sum()/pl.max_horizontal(pl.col('amt').sum(),pl.lit(1.0))).alias(f'{tag}_durw'),
        ((pl.col('amt')*pl.col('pieces')).sum()/pl.max_horizontal(pl.col('amt').sum(),pl.lit(1.0))).alias(f'{tag}_pcw'))
def build(ds):
    out=f'{CACHE}/{ds}.parquet'
    if os.path.exists(out): return ds,'skip'
    try:
        parts=[]
        shp=f'{RAW}/SEL2_TRANSACTION/{ds}.parquet'
        if os.path.exists(shp):
            sh=(pl.scan_parquet(shp)
                .select(pl.col('Symbol').alias('code'),pl.col('TradeAmount').fill_null(0.0).alias('amount'),
                        pl.col('TradePrice').alias('price'),pl.col('UNIX').cast(pl.Float64).alias('ts'),
                        pl.col('BuySellFlag').alias('flag'),pl.col('BuyRecID'),pl.col('SellRecID'))
                .filter(pl.col('code').str.slice(0,2).is_in(list(PREF))))
            tm=sh.group_by('code').agg(pl.col('amount').sum().alias('total_money'))
            s=agg_side(sh.filter(pl.col('flag')=='S').rename({'SellRecID':'oid'}),'oid','as')
            b=agg_side(sh.filter(pl.col('flag')=='B').rename({'BuyRecID':'oid'}),'oid','ab')
            parts.append(tm.join(s,on='code',how='left').join(b,on='code',how='left')
                         .with_columns(pl.lit('SH').alias('exchange')).collect(engine='streaming'))
        szp=f'{RAW}/SZL2_TRADE/{ds}.parquet'
        if os.path.exists(szp):
            sz=(pl.scan_parquet(szp)
                .filter(~(pl.col('TradeType')=='4').fill_null(False))
                .select(pl.col('Symbol').alias('code'),
                        (pl.col('TradePrice')*pl.col('TradeVolume')).fill_null(0.0).alias('amount'),
                        pl.col('TradePrice').alias('price'),pl.col('UNIX').cast(pl.Float64).alias('ts'),
                        pl.col('BuyOrderID'),pl.col('SellOrderID'))
                .filter(pl.col('code').str.slice(0,2).is_in(list(PREF))))
            tm=sz.group_by('code').agg(pl.col('amount').sum().alias('total_money'))
            s=agg_side(sz.filter(pl.col('SellOrderID')>pl.col('BuyOrderID')).rename({'SellOrderID':'oid'}),'oid','as')
            b=agg_side(sz.filter(pl.col('BuyOrderID')>pl.col('SellOrderID')).rename({'BuyOrderID':'oid'}),'oid','ab')
            parts.append(tm.join(s,on='code',how='left').join(b,on='code',how='left')
                         .with_columns(pl.lit('SZ').alias('exchange')).collect(engine='streaming'))
        if not parts: return ds,'nofile'
        pl.concat(parts,how='vertical').with_columns(pl.lit(ds).alias('date')).write_parquet(out+'.tmp')
        os.rename(out+'.tmp',out)
        return ds,'ok'
    except Exception as e:
        return ds,f'ERR:{type(e).__name__}:{e}'
if __name__=='__main__':
    with ProcessPoolExecutor(45) as ex:
        for i,(ds,st) in enumerate(ex.map(build,dates,chunksize=1)):
            if st.startswith('ERR') or i%150==0: print(i,ds,st,flush=True)
    print('traj features done')
