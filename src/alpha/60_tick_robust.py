# -*- coding: utf-8 -*-
"""v31b: 逐笔过滤的稳健性与去重.
1) 滞后窗(t-5..t-1)版 2) 三个近义特征两两组合 3) 是否吸收nl5(去掉nl5看是否掉)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/59_tick_modulate.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
def tfeat_lag(t):
    return tfeat(t,lag=1)
def run_x(tickfeat=None,tickside='excl_high',use_nl5=True,lag=False,second=None,secside='excl_high'):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
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
        selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
        if use_nl5:
            nlv,_,_=feat5(t)
            vv=nlv[dom&~np.isnan(nlv)]
            if len(vv)>200:
                lo=np.nanquantile(vv,1/3)
                selq&=~(np.nan_to_num(nlv,nan=9)<lo)
        fm=dict(zip(FN,tfeat_lag(t) if lag else tfeat(t)))
        for fn,sd in [(tickfeat,tickside),(second,secside)]:
            if fn is None: continue
            v=fm[fn]
            vv2=v[dom&~np.isnan(v)]
            if len(vv2)>200:
                lo2,hi2=np.nanquantile(vv2,[1/3,2/3])
                if sd=='excl_low': selq&=~(np.nan_to_num(v,nan=9)<lo2)
                else: selq&=~(np.nan_to_num(v,nan=-9)>hi2)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=3.0*200) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=200: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        sc2=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc2/=sc2.sum()
        wt=dict(zip(new_h,sc2))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
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
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v21_base']=ev(run_x(),'v21_base')
res['sweep_hi']=ev(run_x('sweep_sell','excl_high'),'sweep_hi')
res['sweep_hi_LAG']=ev(run_x('sweep_sell','excl_high',lag=True),'sweep_hi_LAG')
res['osize_hi']=ev(run_x('osize_sell','excl_high'),'osize_hi')
res['osize_hi_LAG']=ev(run_x('osize_sell','excl_high',lag=True),'osize_hi_LAG')
res['patient_lo']=ev(run_x('patient_sell','excl_low'),'patient_lo')
res['sweep+osize']=ev(run_x('sweep_sell','excl_high',second='osize_sell',secside='excl_high'),'sweep+osize')
res['sweep_hi_NOnl5']=ev(run_x('sweep_sell','excl_high',use_nl5=False),'sweep_hi_NOnl5')
res['osize_hi_NOnl5']=ev(run_x('osize_sell','excl_high',use_nl5=False),'osize_hi_NOnl5')
json.dump(res,open(f'{OUT}/metrics_v31b_robust.json','w'),ensure_ascii=False,indent=1)
print('done')
