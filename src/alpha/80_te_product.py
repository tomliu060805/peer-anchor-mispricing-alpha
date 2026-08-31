# -*- coding: utf-8 -*-
"""v48: TE有向网络三种用法 + 与价格锚的关系诊断"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/78_expB.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
tz=np.load(f'{CACHE}/nets_v47_te.npz')
RBt=tz['t']; NBt=tz['n']; WVt=tz['w']
PG_te=np.full((T,N),np.nan,np.float32); TEQ=np.full((T,N),np.nan,np.float32)
for t in sig5:
    b=np.searchsorted(RBt,t,side='right')-1
    if b<0: continue
    nbr=NBt[b]; wgt=np.maximum(WVt[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-12))[0]
    if len(idx)<100: continue
    nb=nbr[idx]; w0=wgt[idx]
    m=mom20[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=np.maximum(w.sum(1),1e-12)
    mu=(np.nan_to_num(pm)*w).sum(1)/sw
    ok=w.sum(1)>1e-12
    PG_te[t,idx[ok]]=mu[ok]-m[idx[ok]]
    TEQ[t,idx]=w0.mean(1)
# 与价格锚的重合度
ov=[]
for t in sig5[::10]:
    b1=np.searchsorted(RB8,t,side='right')-1; b2=np.searchsorted(RBt,t,side='right')-1
    if b1<0 or b2<0: continue
    a=NB8[b1]; c=NBt[b2]
    valid=(a[:,0]>=0)&(c[:,0]>=0)
    if valid.sum()<100: continue
    inter=(a[valid][:,:,None]==c[valid][:,None,:]).any(2).sum(1)
    ov.append(float(np.mean(inter)/5))
print(f'TE网络与价格网络 top5邻居平均重合率: {np.mean(ov):.3f}',flush=True)
def run10(mode='base',wb=0.4,wh=0.4,ws=0.2,nhold=200):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        g1=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        g2=PG_dual[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        g3=PG_te[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        if mode=='replace': parts=[rank_xs(g3),rank_xs(ZS_b[t]),rank_xs(g2)]
        elif mode=='fourth': parts=[rank_xs(g1),rank_xs(ZS_b[t]),rank_xs(g2),rank_xs(g3)]
        elif mode=='shrink':
            q=TEQ[t]; med=np.nanmedian(q[q>0]) if np.isfinite(q).any() else 1.0
            sc_=np.clip(q/max(med,1e-12),0.3,1.7)
            parts=[rank_xs(g1*sc_),rank_xs(ZS_b[t]),rank_xs(g2*sc_)]
        else: parts=[rank_xs(g1),rank_xs(ZS_b[t]),rank_xs(g2)]
        stk=np.stack(parts)
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        base=np.where(dom,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        fm=dict(zip(FN,tfeat(t)))
        behav_r=np.nanmean(np.stack([rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]),0)
        no_board=((hl5[t]<1)&(ll5[t]<1)).astype(np.float32)
        no_quad=(~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))).astype(np.float32)
        px_ok=(np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)
        struct_r=(no_board+no_quad+px_ok)/3.0
        with np.errstate(all='ignore'):
            comb=wb*rank01(base,hard)+wh*np.nan_to_num(behav_r,nan=0.5)+ws*struct_r
        comb=np.where(hard,comb,np.nan)
        order=np.argsort(-np.nan_to_num(comb,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=3.0*nhold) or np.isnan(comb[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=nhold: break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        w=np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h])
        w=w/w.sum(); wt=dict(zip(new_h,w))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=sum(ww*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,ww in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    return recs
res={}
res['v25_base']=ev2(run10('base'),'v25_base')
res['TE_replace']=ev2(run10('replace'),'TE_replace')
res['TE_fourth']=ev2(run10('fourth'),'TE_fourth')
res['TE_shrink']=ev2(run10('shrink'),'TE_shrink')
json.dump(res,open(f'{OUT}/metrics_v48_te.json','w'),ensure_ascii=False,indent=1)
print('done')
