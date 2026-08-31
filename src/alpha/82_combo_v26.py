# -*- coding: utf-8 -*-
"""v50: 三候选组合验证 (TE第四锚 / ROE gap锚 / dROE恶化过滤). 全组合 + 逐项消融"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/81_fundamental.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
tz=np.load(f'{CACHE}/nets_v47_te.npz')
RBt=tz['t']; NBt=tz['n']; WVt=tz['w']
PG_te=np.full((T,N),np.nan,np.float32)
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
def run12(te=False,roegap=False,droefil=False,wb=0.4,wh=0.4,ws=0.2,nhold=200,track=False):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        g1=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        g2=PG_dual[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        parts=[rank_xs(g1),rank_xs(ZS_b[t]),rank_xs(g2)]
        if te: parts.append(rank_xs(PG_te[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)))
        if roegap: parts.append(rank_xs(fgap(ROE,t)))
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
        sl=[no_board,no_quad,px_ok]
        if droefil:
            dr=DROE[t]; fin=hard&np.isfinite(dr)
            if fin.sum()>200:
                good=(np.nan_to_num(dr,nan=0)>=np.nanquantile(dr[fin],1/3)).astype(np.float32)
            else: good=np.ones(N,np.float32)
            sl.append(good)
        struct_r=np.mean(np.stack(sl),0)
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
def ev3(recs,name):
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
res['v25基线']=ev3(run12(),'v25基线')
res['+TE']=ev3(run12(te=True),'+TE')
res['+ROEgap']=ev3(run12(roegap=True),'+ROEgap')
res['+dROE过滤']=ev3(run12(droefil=True),'+dROE过滤')
res['+TE+ROEgap']=ev3(run12(te=True,roegap=True),'+TE+ROEgap')
res['+TE+dROE']=ev3(run12(te=True,droefil=True),'+TE+dROE')
res['+ROEgap+dROE']=ev3(run12(roegap=True,droefil=True),'+ROEgap+dROE')
res['全部三项']=ev3(run12(te=True,roegap=True,droefil=True),'全部三项')
json.dump(res,open(f'{OUT}/metrics_v50_combo.json','w'),ensure_ascii=False,indent=1)
print('done')
