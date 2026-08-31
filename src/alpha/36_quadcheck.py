# -*- coding: utf-8 -*-
"""v20d: 新口径下 象限过滤(P+S+剔除) 与 触板过滤 的开关对照 (v20配置)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
NHOLD,BM=200,3.0
PG_b,PM_b,DSP_b,ZS_b=S['price']
def run2(quad=True,board=True):
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
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)
        if board: selq&=(hl5[t]<1)&(ll5[t]<1)
        if quad: selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
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
        sc=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc/=sc.sum()
        wt=dict(zip(new_h,sc))
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
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v20_both']=ev(run2(True,True),'v20_both')
res['no_quad']=ev(run2(False,True),'no_quad')
res['no_board']=ev(run2(True,False),'no_board')
res['neither']=ev(run2(False,False),'neither')
json.dump(res,open(f'{OUT}/metrics_v20d_quadcheck.json','w'),ensure_ascii=False,indent=1)
print('done')
