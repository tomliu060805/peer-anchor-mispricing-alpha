# -*- coding: utf-8 -*-
"""v17b: 社区规模统计 + K数量匹配对照 (gap_K20 及 按每股所在社区规模匹配的Km)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np
src=open(PROJ+'/src/alpha/22_four_upgrades.py').read()
head=src.split("print('signals ready'")[0]
exec(head)
# 社区规模统计
sizes_all=[]
for b in range(len(RB21)):
    lab=COMM[b]; ok=lab>=0
    if ok.sum()==0: continue
    sz=np.bincount(lab[ok])
    sz=sz[sz>=6]
    sizes_all.append((len(sz),float(np.mean(sz)),float(np.median(sz)),int(sz.max()),float(ok.mean())))
arr=np.array(sizes_all)
stats={'n_comm_mean':round(float(arr[:,0].mean()),1),'size_mean':round(float(arr[:,1].mean()),1),
 'size_median':round(float(arr[:,2].mean()),1),'size_max_mean':round(float(arr[:,3].mean()),0),
 'coverage_mean':round(float(arr[:,4].mean()),3)}
print('comm stats',json.dumps(stats),flush=True)
# K20 锚 (top20相关邻居, 与社区平均规模同量级)
def gap_k20():
    PG=np.full((T,N),np.nan,np.float32)
    for t in sig_days:
        b=np.searchsorted(RB21,t,side='right')-1
        nbr=N21_20[b]; wgt=np.maximum(W21_20[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wgt[idx]
        m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
        PG[t,idx[ok]]=agg[ok]-m[idx[ok]]
    return PG
PG20=gap_k20()
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
def ic_dom(S,name,mask_to=None):
    d={'dev':[],'val':[]}
    for t in sig_days:
        s=seg_of(t)
        if s is None: continue
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(S[t])
        if mask_to is not None: dom&=~np.isnan(mask_to[t])
        f=np.exp(logc[min(t+HOLD,T-1)]-logc[t])-1
        a=rank_xs(np.where(dom,S[t],np.nan)); y=rank_xs(np.where(dom,f,np.nan))
        m2=~np.isnan(a)&~np.isnan(y)
        if m2.sum()<300: continue
        aa,yy=a[m2],y[m2]
        aa=(aa-aa.mean())/(aa.std()+1e-12); yy=(yy-yy.mean())/(yy.std()+1e-12)
        d[s].append(float((aa*yy).mean()))
    r={s:{'IC':round(float(np.mean(v)),4),'ICIR':round(float(np.mean(v)/(np.std(v)+1e-12)),3),'t':round(tstat(v),2)} for s,v in d.items()}
    print(name,json.dumps(r),flush=True)
    return r
res={'comm_stats':stats}
# 同覆盖对照: 都只在社区覆盖股上评
res['K5_onCommCov']=ic_dom(PG21,'K5_onCommCov',mask_to=PGC)
res['K20_onCommCov']=ic_dom(PG20,'K20_onCommCov',mask_to=PGC)
res['comm_onCommCov']=ic_dom(PGC,'comm_onCommCov')
res['K20_full']=ic_dom(PG20,'K20_full')
json.dump(res,open(f'{OUT}/metrics_v17b_commctrl.json','w'),ensure_ascii=False,indent=1)
print('done')
