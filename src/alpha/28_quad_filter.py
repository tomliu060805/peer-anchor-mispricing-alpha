# -*- coding: utf-8 -*-
"""v19b: v13 + 剔除P+S+象限(去域均同涨掉队) 产品对照."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np
src=open(PROJ+'/src/alpha/26_dsp_guard_elastic.py').read()
head=src.split("# ---- [A]")[0]
exec(head)
z8=np.load(f'{CACHE}/nets_v8.npz'); rb8=z8['rebuilds']; NB8,WV8=z8['nb_p'],z8['wv_p']
sig_days=[int(x) for x in np.arange(int(rb8[0])+1,t_end-7,5)]
PM=np.full((T,N),np.nan,np.float32); PG=np.full((T,N),np.nan,np.float32)
for t in sig_days:
    b=np.searchsorted(rb8,t,side='right')-1
    nbr=NB8[b]; wgt=np.maximum(WV8[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    nb=nbr[idx]; w0=wgt[idx]
    m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=np.maximum(w.sum(1),1e-9)
    mu=(np.nan_to_num(pm)*w).sum(1)/sw
    ok=w.sum(1)>1e-9
    PM[t,idx[ok]]=mu[ok]; PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
ZS=np.full((T,N),np.nan,np.float32)
for b in range(len(rb8)):
    t1=int(rb8[b]); t0=t1-W; t2=min(int(rb8[b+1]) if b+1<len(rb8) else T,T-1)
    sds=[t for t in sig_days if t1<=t<t2]
    if not sds: continue
    nbr=NB8[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
    base=logc[t0-1] if t0>0 else np.zeros(N)
    Pw=np.exp(logc[t0:t1]-base); nbc=np.where(nb>=0,nb,0)
    sj=Pw[:,nbc.ravel()].reshape(W,len(idx),K)
    s_tr=Pw[:,idx][:,:,None]-sj
    mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
    d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
    wsm=np.exp(-d); wsm/=np.maximum(wsm.sum(1,keepdims=True),1e-12)
    for t in sds:
        Pt=np.exp(logc[t]-base)
        zv=(Pt[idx][:,None]-Pt[nbc]-mu)/sd2
        ZS[t,idx]=-np.nansum(zv*wsm,1)
def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
NHOLD,BM=200,3.0
def run(quad_filter=False):
    holdings={}; recs=[]
    for si,t in enumerate(sig_days):
        if si+1>=len(sig_days): break
        a,b2=rank_xs(PG[t]),rank_xs(ZS[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG[t])
        s=np.where(dom,s,np.nan)
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        if quad_filter:
            dm=np.nanmean(np.where(dom,mom[t],np.nan))
            pp=(PM[t]-dm>=0)&(mom[t]-dm>=0)
            selq&=~pp
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
res['base_v13']=ev(run(),'base_v13')
res['quad_filter']=ev(run(quad_filter=True),'quad_filter')
json.dump(res,open(f'{OUT}/metrics_v19b_quadfilter.json','w'),ensure_ascii=False,indent=1)
print('done')
