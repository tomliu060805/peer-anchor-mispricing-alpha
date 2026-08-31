# -*- coding: utf-8 -*-
"""v25: 六个小盘定价误差过滤候选, v20配置逐个加装对照.
F1次新(上市<250日) F2低价(<2元) F3流动性(5日ADV域内最低10%) F4事件急跌(近5日单日跌>7%且当日全指跌<1%)
F5解禁代理(20日内流通股本增>5%) F6慢阴跌(20日下跌天数>=14)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, STOCK_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
close_raw=g['close']; money_g=g['money']
code_ix={c:i for i,c in enumerate(codes)}
# 上市日
si=pd.read_parquet(STOCK_ROOT+'/info/stock_info.parquet')
start=np.full(N,np.datetime64('1990-01-01'))
for c,sd in zip(si['code'],si['start_date']):
    i=code_ix.get(c,-1)
    if i>=0: start[i]=np.datetime64(str(sd)[:10])
age_ok=np.zeros((T,N),bool)
for t in range(T): age_ok[t]=(dnum[t]-start).astype('timedelta64[D]').astype(int)>=365
# 流通股本(解禁代理)
vroot=STOCK_ROOT+'/fundamental/valuation'
def load_cc(i):
    f=f'{vroot}/{dates[i]}.parquet'
    out=np.full(N,np.nan,np.float32)
    if not os.path.exists(f): return i,out
    d=pd.read_parquet(f,columns=['code','circulating_cap'])
    ix=np.array([code_ix.get(c,-1) for c in d['code']])
    v=d['circulating_cap'].values.astype(np.float32); okm=ix>=0
    out[ix[okm]]=v[okm]
    return i,out
if not os.path.exists(f'{CACHE}/circcap_grid.npy'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(load_cc,range(T),chunksize=16))
    CC=np.full((T,N),np.nan,np.float32)
    for i,o in outs: CC[i]=o
    np.save(f'{CACHE}/circcap_grid.npy',CC)
CC=np.load(f'{CACHE}/circcap_grid.npy')
with np.errstate(all='ignore'):
    cc_jump=np.zeros((T,N),bool)
    cc_jump[20:]=(CC[20:]/CC[:-20]-1)>0.05
# 5日ADV
cm=np.cumsum(np.nan_to_num(money_g,nan=0.0),0)
adv5=np.full((T,N),np.nan,np.float32); adv5[5:]=(cm[5:]-cm[:-5])/5
# 事件急跌: 单日跌>7%且全指当日跌<1%, 近5日内
evt=(np.nan_to_num(ret,nan=0)< -0.07)&(qz[:,None]>-0.01)
evt5=np.zeros((T,N),bool)
ce=np.cumsum(evt.astype(np.int32),0)
evt5[5:]=(ce[5:]-ce[:-5])>0
# 慢阴跌: 20日内下跌天数
dn=(np.nan_to_num(ret,nan=0)<0).astype(np.int32); cd=np.cumsum(dn,0)
down20=np.zeros((T,N),np.int32); down20[20:]=cd[20:]-cd[:-20]
NHOLD,BM=200,3.0
PG_b,PM_b,DSP_b,ZS_b=S['price']
def run_f(extra=None):
    holdings={}; recs=[]
    for si2,t in enumerate(sig5):
        if si2+1>=len(sig5): break
        gap=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZS_b[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
        if extra=='F1': selq&=age_ok[t]
        elif extra=='F2': selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
        elif extra=='F3':
            v=adv5[t]; vv=v[dom&~np.isnan(v)]
            if len(vv)>200: selq&=np.nan_to_num(v,nan=-9)>=np.nanquantile(vv,0.10)
        elif extra=='F4': selq&=~evt5[t]
        elif extra=='F5': selq&=~cc_jump[t]
        elif extra=='F6': selq&=down20[t]<14
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=BM*NHOLD) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        sc2=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc2/=sc2.sum()
        wt=dict(zip(new_h,sc2))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si2+1]
        pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq)); holdings=wt
    return recs
def ev(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['base']=ev(run_f(None),'base')
for f in ['F1','F2','F3','F4','F5','F6']:
    res[f]=ev(run_f(f),f)
json.dump(res,open(f'{OUT}/metrics_v25_scfilters.json','w'),ensure_ascii=False,indent=1)
print('done')
