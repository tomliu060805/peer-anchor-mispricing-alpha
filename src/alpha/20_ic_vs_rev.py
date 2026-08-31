# -*- coding: utf-8 -*-
"""v15: 定型合成因子(活跃域) 周频IC; 与纯反转-mom20 的相关/双向正交残差IC."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np
CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM,HOLD=120,5,20,250,5
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')
g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); ll=np.nan_to_num(at_ll,nan=0.0)
chl=np.cumsum(hl,0); cll=np.cumsum(ll,0)
lim250=np.zeros((T,N),np.float32); lim250[WLIM:]=chl[WLIM:]-chl[:-WLIM]
hl5=np.zeros((T,N),np.float32); hl5[5:]=chl[5:]-chl[:-5]
ll5=np.zeros((T,N),np.float32); ll5[5:]=cll[5:]-cll[:-5]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
t_start=int(rb[0])+1; t_end=int(np.searchsorted(dnum,END,side='right'))
sig_days=np.arange(t_start,t_end-6,5)

PG=np.full((T,N),np.nan,np.float32); ZS=np.full((T,N),np.nan,np.float32)
for t in sig_days:
    b=np.searchsorted(rb,t,side='right')-1
    nbr=NB_P[b]; wgt=np.maximum(WV_P[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    nb=nbr[idx]; w0=wgt[idx]
    m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=w.sum(1); ok=sw>1e-9
    agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
    PG[t,idx[ok]]=agg[ok]-m[idx[ok]]
for b in range(len(rb)):
    t1=rb[b]; t0=t1-W; t2=min(rb[b+1] if b+1<len(rb) else T,T-1)
    sds=[t for t in sig_days if t1<=t<t2]
    if not sds: continue
    nbr=NB_P[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
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
def seg_of(t):
    if dnum[t]>=np.datetime64(DEV[0]) and dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]>=np.datetime64(VAL[0]) and dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))
acc={s:{k:[] for k in ['ic_combo','ic_rev','ic_combo_orth','ic_rev_orth','corr','ic_combo_prod']} for s in ['dev','val']}
for t in sig_days:
    s=seg_of(t)
    if s is None: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    a=rank_xs(np.where(dom,PG[t],np.nan)); b2=rank_xs(np.where(dom,ZS[t],np.nan))
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        combo=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
    rev=np.where(dom,-mom[t],np.nan)
    f=np.exp(logc[min(t+HOLD,T-1)]-logc[t])-1
    cr=rank_xs(combo); rr=rank_xs(rev); fr=rank_xs(np.where(dom,f,np.nan))
    m2=~np.isnan(cr)&~np.isnan(rr)&~np.isnan(fr)
    if m2.sum()<300: continue
    c,r0,y=cr[m2],rr[m2],fr[m2]
    c=(c-c.mean())/(c.std()+1e-12); r0=(r0-r0.mean())/(r0.std()+1e-12); y=(y-y.mean())/(y.std()+1e-12)
    acc[s]['corr'].append(float((c*r0).mean()))
    acc[s]['ic_combo'].append(float((c*y).mean()))
    acc[s]['ic_rev'].append(float((r0*y).mean()))
    co=c-(c*r0).mean()*r0; co/= (co.std()+1e-12)
    ro=r0-(c*r0).mean()*c; ro/= (ro.std()+1e-12)
    acc[s]['ic_combo_orth'].append(float((co*y).mean()))
    acc[s]['ic_rev_orth'].append(float((ro*y).mean()))
    # 产品版(触板过滤后)的IC
    dom2=dom&(hl5[t]<1)&(ll5[t]<1)
    cr2=rank_xs(np.where(dom2,combo,np.nan)); fr2=rank_xs(np.where(dom2,f,np.nan))
    m3=~np.isnan(cr2)&~np.isnan(fr2)
    if m3.sum()>300:
        c2,y2=cr2[m3],fr2[m3]
        c2=(c2-c2.mean())/(c2.std()+1e-12); y2=(y2-y2.mean())/(y2.std()+1e-12)
        acc[s]['ic_combo_prod'].append(float((c2*y2).mean()))
res={}
for s in ['dev','val']:
    res[s]={k:{'mean':round(float(np.nanmean(v)),4),'ICIR':round(float(np.nanmean(v)/(np.nanstd(v)+1e-12)),3),
               't':round(tstat(v),2),'n':len(v)} for k,v in acc[s].items()}
json.dump(res,open(f'{OUT}/metrics_v15_ic_rev.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
