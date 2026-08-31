# -*- coding: utf-8 -*-
"""v19c: 空头信息用于退出提速 — 持仓股进入P-S+状态(自己逆势涨而兄弟在跌)立即卖, 其余同v19."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np
src=open(PROJ+'/src/alpha/28_quad_filter.py').read()
head=src.split("def run(")[0]
exec(head)
NHOLD,BM=200,3.0
def run(quad_filter=True,exit_accel=False):
    holdings={}; recs=[]
    for si,t in enumerate(sig_days):
        if si+1>=len(sig_days): break
        a,b2=rank_xs(PG[t]),rank_xs(ZS[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        if quad_filter:
            pp=(PM[t]-dm>=0)&(mom[t]-dm>=0)
            selq&=~pp
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            sell=(rank[i2]>=BM*NHOLD) or np.isnan(s[i2])
            if exit_accel and PM[t,i2]==PM[t,i2] and mom[t,i2]==mom[t,i2]:
                if (PM[t,i2]-dm<0) and (mom[t,i2]-dm>=0): sell=True
            if sell and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<50: holdings={}; continue
        wt={i2:1.0/len(new_h) for i2 in new_h}
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig_days[si+1]
        pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    return recs
def ev(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/HOLD
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',('2015-01-01','2024-08-16'))]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
          'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'turn':round(float(tn[m].mean()),3)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v19']=ev(run(),'v19')
res['v19_exit_accel']=ev(run(exit_accel=True),'v19_exit_accel')
json.dump(res,open(f'{OUT}/metrics_v19c_exit.json','w'),ensure_ascii=False,indent=1)
print('done')
