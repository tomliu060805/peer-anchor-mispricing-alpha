# -*- coding: utf-8 -*-
"""v19: gap的四象限拆解 (peer_mom符号 × own_mom符号).
口径1: 原始符号; 口径2: 去域均后的符号(剔市场行情影响).
输出: 各象限内 gap IC / 多头腿(全域top10% gap)落在该象限的 fwd超额与占比."""
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
hl=np.nan_to_num(at_hl,nan=0.0); chl=np.cumsum(hl,0)
lim250=np.zeros((T,N),np.float32); lim250[WLIM:]=chl[WLIM:]-chl[:-WLIM]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
t_start=int(rb[0])+1; t_end=int(np.searchsorted(dnum,END,side='right'))
sig_days=[int(x) for x in np.arange(t_start,t_end-6,5)]
PM=np.full((T,N),np.nan,np.float32); PG=np.full((T,N),np.nan,np.float32)
for t in sig_days:
    b=np.searchsorted(rb,t,side='right')-1
    nbr=NB_P[b]; wgt=np.maximum(WV_P[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    nb=nbr[idx]; w0=wgt[idx]
    m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=np.maximum(w.sum(1),1e-9)
    mu=(np.nan_to_num(pm)*w).sum(1)/sw
    ok=w.sum(1)>1e-9
    PM[t,idx[ok]]=mu[ok]; PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
def seg_of(t):
    if dnum[t]>=np.datetime64(DEV[0]) and dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]>=np.datetime64(VAL[0]) and dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))
def ic_of(x,y):
    xr=np.argsort(np.argsort(x)).astype(float); yr=np.argsort(np.argsort(y)).astype(float)
    xr=(xr-xr.mean())/(xr.std()+1e-12); yr=(yr-yr.mean())/(yr.std()+1e-12)
    return float((xr*yr).mean())
QN=['P+S+','P+S-','P-S-','P-S+']
res={}
for mode in ['raw','demean']:
    ic_acc={s:{q:[] for q in QN} for s in ['dev','val']}
    lg_acc={s:{q:{'ex':[],'cnt':0} for q in QN} for s in ['dev','val']}
    tot_cnt={s:0 for s in ['dev','val']}
    for t in sig_days:
        s=seg_of(t)
        if s is None: continue
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG[t])
        ix=np.where(dom)[0]
        if len(ix)<400: continue
        pm=PM[t,ix].astype(float); om=mom[t,ix].astype(float)
        if mode=='demean':
            dm=np.nanmean(mom[t,ix]); pm=pm-dm; om=om-dm
        gv=PG[t,ix]; f=np.exp(logc[min(t+HOLD,T-1),ix]-logc[t,ix])-1
        fu=np.nanmean(f)
        quad=np.where(pm>=0,np.where(om>=0,0,1),np.where(om<0,2,3))
        for qi,q in enumerate(QN):
            m2=quad==qi
            if m2.sum()>=60: ic_acc[s][q].append(ic_of(gv[m2],f[m2]))
        thr=np.quantile(gv,0.9); top=gv>=thr
        tot_cnt[s]+=int(top.sum())
        for qi,q in enumerate(QN):
            m2=top&(quad==qi)
            lg_acc[s][q]['cnt']+=int(m2.sum())
            if m2.sum()>=3: lg_acc[s][q]['ex'].append(float(np.nanmean(f[m2])-fu))
    for s in ['dev','val']:
        res[f'{mode}_{s}']={}
        for q in QN:
            v=ic_acc[s][q]; e=lg_acc[s][q]['ex']
            res[f'{mode}_{s}'][q]={
              'IC':round(float(np.mean(v)),4) if len(v)>20 else None,
              'IC_t':round(tstat(v),2) if len(v)>20 else None,
              'long_ex_bp':round(float(np.mean(e))*1e4,1) if len(e)>20 else None,
              'long_t':round(tstat(e),2) if len(e)>20 else None,
              'long_share%':round(lg_acc[s][q]['cnt']/max(tot_cnt[s],1)*100,1)}
json.dump(res,open(f'{OUT}/metrics_v19_quadrant.json','w'),ensure_ascii=False,indent=1)
for k,v in res.items(): print(k,json.dumps(v,ensure_ascii=False))
