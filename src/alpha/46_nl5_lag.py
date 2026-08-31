# -*- coding: utf-8 -*-
"""v24d: nl5过滤的滞后窗稳健性 (t-5..t-1, 不含当日, 实时完全可得)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/45_l2_modulate.py').read()
head=src.split("# A) 条件表")[0]
exec(head)
def feat5_lag(t):
    s=mfc[t-1]-mfc[t-6]
    lb,ls,tb,ts,tm=s
    tm=np.maximum(tm,1.0)
    nl=(lb-ls)/tm
    has=(mfc[t-1,4]-mfc[t-6,4])>0
    return np.where(has,nl,np.nan)
NHOLD,BM=200,3.0
def run_lag(use_filter):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5): break
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
        if use_filter:
            v=feat5_lag(t)
            vv=v[dom&~np.isnan(v)]
            if len(vv)>200:
                lo=np.nanquantile(vv,1/3)
                selq&=~(np.nan_to_num(v,nan=9)<lo)
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
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2)}
    m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
    years=sorted(set(str(d)[:4] for d in ds[m]))
    r['yearly']={y:round(float(np.prod(1+ex[m][[str(x)[:4]==y for x in ds[m]]])-1),4) for y in years}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['base']=ev(run_lag(False),'base')
res['nl5_lag']=ev(run_lag(True),'nl5_lag')
json.dump(res,open(f'{OUT}/metrics_v24d_nl5lag.json','w'),ensure_ascii=False,indent=1)
print('done')
