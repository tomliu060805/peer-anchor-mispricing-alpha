# -*- coding: utf-8 -*-
"""v28: 隐性行业专属版. A纯跨行业网络 B纯同行业网络(对照) C选股层过滤(frac_same<=0.2). v21配置."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/48_combo_final.py').read()
head=src.split("res={}")[0]
exec(head)
close_raw=g['close']
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8=z8['nb_p']
rb_list=[int(x) for x in RB8]
def build_masked(args):
    t1,mode=args
    Rw=ret[t1-W:t1]
    valid=(~np.isnan(Rw)).sum(0)>=110
    valid&=~(st_g[t1-1]==1)
    dstr=str(dates[t1-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    inds=np.array([imap.get(c,'') for c in codes])
    valid&=inds!=''
    idx=np.where(valid)[0]; n=len(idx)
    X=np.nan_to_num(Rw[:,idx],nan=0.0).astype(np.float64)
    mkt=X.mean(1,keepdims=True)
    beta=(X*mkt).sum(0)/np.maximum((mkt*mkt).sum(),1e-12)
    Xr=X-mkt@beta[None,:]
    gi=inds[idx]
    for u in np.unique(gi):
        sel=gi==u
        if sel.sum()>1: Xr[:,sel]-=Xr[:,sel].mean(1,keepdims=True)
    sd=Xr.std(0)+1e-12; Z=(Xr-Xr.mean(0))/sd
    C=(Z.T@Z/W).astype(np.float32); np.fill_diagonal(C,-9)
    same=(gi[:,None]==gi[None,:])
    if mode=='cross': C[same]=-9
    else: C[~same]=-9
    part=np.argpartition(-C,K,axis=1)[:,:K]
    rows=np.arange(n)[:,None]; vals=C[rows,part]
    o=np.argsort(-vals,axis=1)
    nb,wv=part[rows,o],vals[rows,o]
    wv=np.where(wv>-8,wv,0)   # 无效边(被禁)权重0
    on=np.full((N,K),-1,np.int32); ow=np.zeros((N,K),np.float32)
    on[idx]=np.where(wv>0,idx[nb],-1); ow[idx]=np.maximum(wv,0)
    return t1,on,ow
if not os.path.exists(f'{CACHE}/nets_v28_ind.npz'):
    store={}
    for mode in ['cross','same']:
        with ProcessPoolExecutor(50) as ex:
            outs=list(ex.map(build_masked,[(t,mode) for t in rb_list],chunksize=1))
        outs.sort(key=lambda x:x[0])
        store[f'{mode}_n']=np.stack([o[1] for o in outs]); store[f'{mode}_w']=np.stack([o[2] for o in outs])
        print(f'{mode} net built',flush=True)
    np.savez_compressed(f'{CACHE}/nets_v28_ind.npz',**store)
nz=np.load(f'{CACHE}/nets_v28_ind.npz')
def make_pg_zs(NB,WV):
    PG=np.full((T,N),np.nan,np.float32); PM=np.full((T,N),np.nan,np.float32)
    for t in sig5:
        b=np.searchsorted(RB8,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wgt[idx]
        m=mom20[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=np.maximum(w.sum(1),1e-9)
        mu=(np.nan_to_num(pm)*w).sum(1)/sw
        ok=w.sum(1)>1e-9
        PG[t,idx[ok]]=mu[ok]-m[idx[ok]]; PM[t,idx[ok]]=mu[ok]
    ZS=np.full((T,N),np.nan,np.float32)
    for b in range(len(RB8)):
        t1=int(RB8[b]); t0=t1-W; t2=min(int(RB8[b+1]) if b+1<len(RB8) else T,T-1)
        sds=[t for t in sig5 if t1<=t<t2]
        if not sds: continue
        nbr=NB[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
        base=logc[t0-1] if t0>0 else np.zeros(N)
        Pw=np.exp(logc[t0:t1]-base); nbc=np.where(nb>=0,nb,0)
        sj=Pw[:,nbc.ravel()].reshape(W,len(idx),K)
        s_tr=Pw[:,idx][:,:,None]-sj
        mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
        d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
        wsm=np.exp(-d)*(nb>=0); wsm/=np.maximum(wsm.sum(1,keepdims=True),1e-12)
        for t in sds:
            Pt=np.exp(logc[t]-base)
            zv=(Pt[idx][:,None]-Pt[nbc]-mu)/sd2
            ZS[t,idx]=-np.nansum(zv*wsm,1)
    return PG,PM,ZS
NETS={'base':(PG_b,PM_b,ZS_b)}
for mode in ['cross','same']:
    NETS[mode]=make_pg_zs(nz[f'{mode}_n'],nz[f'{mode}_w'])
print('signals ready',flush=True)
# frac_same (基线网络)
def frac_same_at(t):
    b=np.searchsorted(RB8,t,side='right')-1
    nbr=NB8[b]
    dstr=str(dates[t-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    inds=np.array([imap.get(c,'') for c in codes])
    ok=nbr>=0
    same=(inds[np.where(ok,nbr,0)]==inds[:,None])&ok
    return same.sum(1)/np.maximum(ok.sum(1),1)
def run_v(sigkey,fs_filter=False):
    PGx,PMx,ZSx=NETS[sigkey]
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5): break
        gap=PGx[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZSx[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PGx[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((PMx[t]-dm>=0)&(mom20[t]-dm>=0))
        selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
        nlv,_,_=feat5(t)
        vv=nlv[dom&~np.isnan(nlv)]
        if len(vv)>200:
            lo=np.nanquantile(vv,1/3)
            selq&=~(np.nan_to_num(nlv,nan=9)<lo)
        if fs_filter:
            selq&=frac_same_at(t)<=0.2
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
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v21_base']=ev(run_v('base'),'v21_base')
res['net_crossonly']=ev(run_v('cross'),'net_crossonly')
res['net_sameonly']=ev(run_v('same'),'net_sameonly')
res['sel_crossfilter']=ev(run_v('base',fs_filter=True),'sel_crossfilter')
json.dump(res,open(f'{OUT}/metrics_v28_crossind.json','w'),ensure_ascii=False,indent=1)
print('done')
