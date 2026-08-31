# -*- coding: utf-8 -*-
"""v45: 轨迹特征产品测试. 两个两段单调赢家: split_s(拆单卖,越多越差) imb_fast(买卖急迫失衡,越高越好)
加入行为块, 与v25基线对照"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/76_traj_analysis.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
nzf=np.load(f'{CACHE}/nets_v41_flow.npz')
def make_gap_dual():
    PG=np.full((T,N),np.nan,np.float32)
    RBx=nzf['dual_t']; NB=nzf['dual_n']; WV=nzf['dual_w']
    for t in sig5:
        b=np.searchsorted(RBx,t,side='right')-1
        if b<0: continue
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        if len(idx)<100: continue
        nb=nbr[idx]; w0=wgt[idx]
        m=mom20[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=np.maximum(w.sum(1),1e-9)
        mu=(np.nan_to_num(pm)*w).sum(1)/sw
        ok=w.sum(1)>1e-9
        PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
    return PG
PG_dual=make_gap_dual()
def run8(extra=(),wb=0.4,wh=0.4,ws=0.2,nhold=200):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        g1=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        g2=PG_dual[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        stk=np.stack([rank_xs(g1),rank_xs(ZS_b[t]),rank_xs(g2)])
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        base=np.where(dom,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        fm=dict(zip(FN,tfeat(t))); tj=tj_feat(t)
        bl=[rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]
        for e in extra:
            if e=='split': bl.append(rank01(-tj['split_s'],hard))
            elif e=='imbfast': bl.append(rank01(tj['imb_fast'],hard))
            elif e=='fast': bl.append(rank01(-tj['fast_s'],hard))
        behav_r=np.nanmean(np.stack(bl),0)
        no_board=((hl5[t]<1)&(ll5[t]<1)).astype(np.float32)
        no_quad=(~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))).astype(np.float32)
        px_ok=(np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)
        struct_r=(no_board+no_quad+px_ok)/3.0
        with np.errstate(all='ignore'):
            comb=wb*base_r_calc(base,hard)+wh*np.nan_to_num(behav_r,nan=0.5)+ws*struct_r
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
def base_r_calc(base,hard): return rank01(base,hard)
def ev2(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'turn':round(float(tn[m].mean()),3)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v25_base']=ev2(run8(()),'v25_base')
res['+split']=ev2(run8(('split',)),'+split')
res['+imbfast']=ev2(run8(('imbfast',)),'+imbfast')
res['+fast']=ev2(run8(('fast',)),'+fast')
res['+split+imbfast']=ev2(run8(('split','imbfast')),'+split+imbfast')
res['+split+imbfast+fast']=ev2(run8(('split','imbfast','fast')),'+split+imbfast+fast')
json.dump(res,open(f'{OUT}/metrics_v45_trajprod.json','w'),ensure_ascii=False,indent=1)
print('done')
