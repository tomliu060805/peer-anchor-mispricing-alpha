# -*- coding: utf-8 -*-
"""v43(方向4): 修复完成度卖出. 进场记gap0, 当 gap_t/gap0 < thr 即卖(绝对退出), 与排名规则对照/叠加."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/68_behav_decay.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
def run6(mode='rank',thr=0.3,nhold=200,wb=0.4,wh=0.4,ws=0.2):
    holdings={}; gap0={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        gapv=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gapv),rank_xs(ZS_b[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        base=np.where(dom,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; gap0={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        fm=dict(zip(FN,tfeat(t)))
        base_r=rank01(base,hard)
        behav_r=np.nanmean(np.stack([rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]),0)
        no_board=((hl5[t]<1)&(ll5[t]<1)).astype(np.float32)
        no_quad=(~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))).astype(np.float32)
        px_ok=(np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)
        struct_r=(no_board+no_quad+px_ok)/3.0
        with np.errstate(all='ignore'):
            comb=wb*base_r+wh*np.nan_to_num(behav_r,nan=0.5)+ws*struct_r
        comb=np.where(hard,comb,np.nan)
        order=np.argsort(-np.nan_to_num(comb,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            sell=False
            if mode in ('rank','both'):
                if (rank[i2]>=3.0*nhold) or np.isnan(comb[i2]): sell=True
            if mode in ('repair','both'):
                g0=gap0.get(i2,np.nan); gt=PG_b[t,i2]
                if g0==g0 and g0>1e-6 and gt==gt and gt/g0<thr: sell=True
                if np.isnan(comb[i2]): sell=True
            if sell and can_sell[i2]:
                del new_h[i2]; gap0.pop(i2,None)
        for i2 in order:
            if len(new_h)>=nhold: break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2]: continue
            new_h[i2]=0.0; gap0[i2]=float(PG_b[t,i2])
        if len(new_h)<40: holdings={}; gap0={}; continue
        w=np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h])
        w=w/w.sum(); wt=dict(zip(new_h,w))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=sum(ww*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,ww in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    return recs
def ev_full(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'turn':round(float(tn[m].mean()),3)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v24_排名规则']=ev_full(run6('rank'),'v24_排名规则')
for thr in [0.2,0.3,0.5]:
    res[f'repair_thr{thr}']=ev_full(run6('repair',thr),f'repair_thr{thr}')
    res[f'both_thr{thr}']=ev_full(run6('both',thr),f'both_thr{thr}')
json.dump(res,open(f'{OUT}/metrics_v43_exit.json','w'),ensure_ascii=False,indent=1)
print('done')
