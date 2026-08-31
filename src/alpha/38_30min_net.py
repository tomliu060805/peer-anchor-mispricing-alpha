# -*- coding: utf-8 -*-
"""v21: 30分钟收益联动网络. 日内bar-to-bar收益(7/日), 残差corr top-5,
窗口 20d/40d/120d 三变体, 下游v20产品不变只换邻居. 对照日频120d基线."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, STOCK_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM,COST=120,5,20,250,0.0010
DEV=('2016-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); FULL=('2016-01-01','2024-08-16')
END=np.datetime64('2024-08-16')
g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
code_ix={c:i for i,c in enumerate(codes)}
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st_g=np.load(f'{CACHE}/st_grid.npz')['is_st']
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom20=np.full((T,N),np.nan,np.float32); mom20[20:]=(logc[20:]-logc[:-20]).astype(np.float32)
r2=np.nan_to_num(ret,nan=0.0)**2; c2=np.cumsum(r2,0)
vol20=np.full((T,N),np.nan,np.float32); vol20[20:]=np.sqrt((c2[20:]-c2[:-20])/20).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st_g<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); llm=np.nan_to_num(at_ll,nan=0.0)
chl=np.cumsum(hl,0); cll=np.cumsum(llm,0)
lim250=np.zeros((T,N),np.float32); lim250[WLIM:]=chl[WLIM:]-chl[:-WLIM]
hl5=np.zeros((T,N),np.float32); hl5[5:]=chl[5:]-chl[:-5]
ll5=np.zeros((T,N),np.float32); ll5[5:]=cll[5:]-cll[:-5]
qz=np.load(f'{CACHE}/quanzhi_ret.npy'); lgq=np.cumsum(np.log1p(qz))
t_end=int(np.searchsorted(dnum,END,side='right'))

def load_day30(i):
    f=f'{STOCK_ROOT}/price/price_30m/{dates[i]}.parquet'
    out=np.full((8,N),np.nan,np.float32)
    if not os.path.exists(f): return i,out
    d=pd.read_parquet(f,columns=['datetime','code','close'])
    d=d.sort_values('datetime')
    bars=sorted(d['datetime'].unique())[:8]
    bmap={b:j for j,b in enumerate(bars)}
    ix=np.array([code_ix.get(c,-1) for c in d['code']])
    bj=np.array([bmap.get(b,-1) for b in d['datetime']])
    v=d['close'].values.astype(np.float32)
    okm=(ix>=0)&(bj>=0)
    out[bj[okm],ix[okm]]=v[okm]
    return i,out
if not os.path.exists(f'{CACHE}/ret30_grid.npz'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(load_day30,range(T),chunksize=8))
    C30=np.full((T,8,N),np.nan,np.float32)
    for i,o in outs: C30[i]=o
    with np.errstate(all='ignore'):
        R30=np.log(C30[:,1:,:]/C30[:,:-1,:]).astype(np.float32)   # (T,7,N) 日内bar收益
    R30=np.where(np.isfinite(R30),R30,np.nan)
    np.savez_compressed(f'{CACHE}/ret30_grid.npz',r=R30)
    print('ret30 cached',flush=True)
R30=np.load(f'{CACHE}/ret30_grid.npz')['r']
print('R30 loaded',R30.shape,flush=True)

rb21=list(range(WLIM,T,21))
def build30(args):
    t1,wd=args
    seg=R30[t1-wd:t1].reshape(wd*7,N)
    valid=(~np.isnan(seg)).sum(0)>=int(wd*7*0.85)
    valid&=~(st_g[t1-1]==1)
    dstr=str(dates[t1-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    inds=np.array([imap.get(c,'') for c in codes])
    valid&=inds!=''
    idx=np.where(valid)[0]; n=len(idx)
    X=np.nan_to_num(seg[:,idx],nan=0.0).astype(np.float64)
    mkt=X.mean(1,keepdims=True)
    beta=(X*mkt).sum(0)/np.maximum((mkt*mkt).sum(),1e-12)
    Xr=X-mkt@beta[None,:]
    gi=inds[idx]
    for u in np.unique(gi):
        sel=gi==u
        if sel.sum()>1: Xr[:,sel]-=Xr[:,sel].mean(1,keepdims=True)
    sd=Xr.std(0)+1e-12; Z=(Xr-Xr.mean(0))/sd
    C=(Z.T@Z/len(Z)).astype(np.float32); np.fill_diagonal(C,-9)
    part=np.argpartition(-C,K,axis=1)[:,:K]
    rows=np.arange(n)[:,None]; vals=C[rows,part]
    o=np.argsort(-vals,axis=1)
    on=np.full((N,K),-1,np.int32); ow=np.zeros((N,K),np.float32)
    on[idx]=idx[part[rows,o]]; ow[idx]=vals[rows,o]
    return t1,on,ow
if not os.path.exists(f'{CACHE}/nets_v21_30m.npz'):
    store={}
    for wd in [20,40,120]:
        with ProcessPoolExecutor(40) as ex:
            outs=list(ex.map(build30,[(t,wd) for t in rb21],chunksize=1))
        outs.sort(key=lambda x:x[0])
        store[f'w{wd}_n']=np.stack([o[1] for o in outs]); store[f'w{wd}_w']=np.stack([o[2] for o in outs])
        print(f'w{wd} built',flush=True)
    np.savez_compressed(f'{CACHE}/nets_v21_30m.npz',rb=np.array(rb21,np.int32),**store)
n21=np.load(f'{CACHE}/nets_v21_30m.npz')
z8=np.load(f'{CACHE}/nets_v8.npz')
RB=np.array(rb21,np.int32)
sig5=[int(x) for x in np.arange(int(z8['rebuilds'][0])+1,t_end-7,5)]
def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
def make_sig(RBx,NB,WV):
    PG=np.full((T,N),np.nan,np.float32)
    for t in sig5:
        b=np.searchsorted(RBx,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wgt[idx]
        m=mom20[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=np.maximum(w.sum(1),1e-9)
        mu=(np.nan_to_num(pm)*w).sum(1)/sw
        ok=w.sum(1)>1e-9
        PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
    ZS=np.full((T,N),np.nan,np.float32)
    for b in range(len(RBx)):
        t1=int(RBx[b]); t0=t1-W; t2=min(int(RBx[b+1]) if b+1<len(RBx) else T,T-1)
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
    return PG,ZS
SIGS={}
SIGS['daily120']=make_sig(z8['rebuilds'],z8['nb_p'],z8['wv_p'])
for wd in [20,40,120]:
    SIGS[f'm30_w{wd}']=make_sig(RB,n21[f'w{wd}_n'],n21[f'w{wd}_w'])
print('signals ready',flush=True)
NHOLD,BM=200,3.0
def run(PGx,ZSx):
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
        # PM 需按当前网络: 用 gap 反推 pm = gap+mom
        pmv=PGx[t]+mom20[t]
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((pmv-dm>=0)&(mom20[t]-dm>=0))
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
for nm,(PGx,ZSx) in SIGS.items():
    res[nm]=ev(run(PGx,ZSx),nm)
# 混合: 日频网络gap + 30min w40网络gap 秩平均
PA,ZA=SIGS['daily120']; PB,_=SIGS['m30_w40']
PMIX=np.full((T,N),np.nan,np.float32)
for t in sig5:
    a,b2=rank_xs(PA[t]),rank_xs(PB[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        PMIX[t]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
res['mix_daily_m30w40']=ev(run(PMIX,ZA),'mix_daily_m30w40')
json.dump(res,open(f'{OUT}/metrics_v21_30min.json','w'),ensure_ascii=False,indent=1)
print('done')
