# -*- coding: utf-8 -*-
"""v27: 网络的"隐性行业"属性检验.
1) top5邻居同申万一级占比(逐期)
2) 多头腿按"邻居跨行业比例"分组的fwd超额: 跨行业锚是否更有效"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, pickle, numpy as np
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8=z8['nb_p']
PG_b,PM_b,DSP_b,ZS_b=S['price']
def ind_of(t):
    dstr=str(dates[t-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    return np.array([imap.get(c,'') for c in codes])
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
same_frac=[]
cond={s:{q:[] for q in range(3)} for s in ['dev','val']}
for t in sig5:
    s0=seg_of(t)
    if s0 is None: continue
    b=np.searchsorted(RB8,t,side='right')-1
    nbr=NB8[b]
    inds=ind_of(t)
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    nb=nbr[ix]
    ok=nb>=0
    same=(inds[np.where(nb>=0,nb,0)]==inds[ix][:,None])&ok
    frac_same=same.sum(1)/np.maximum(ok.sum(1),1)
    same_frac.append(float(np.nanmean(frac_same)))
    # 多头腿按跨行业比例三分组
    gap=PG_b[t,ix]/np.maximum(vol20[t,ix]*np.sqrt(20),1e-4)
    a,b2=rank_xs(PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)),rank_xs(ZS_b[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        sc=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))[ix]
    f=np.exp(logc[min(t+5,T-1),ix]-logc[t,ix])-1
    fu=np.nanmean(f)
    thr=np.nanquantile(sc,0.9); top=sc>=thr
    cross=1-frac_same
    qe=np.nanquantile(cross[top],[1/3,2/3])
    qq=np.digitize(cross[top],qe)
    for q in range(3):
        mm=(qq==q)
        if mm.sum()>=5: cond[s0][q].append(float(np.nanmean(f[top][mm])-fu))
res={'same_industry_frac_mean':round(float(np.mean(same_frac)),3)}
for s0 in ['dev','val']:
    res[f'longleg_by_cross_{s0}']={f'q{q}(跨行业{"低中高"[q]})':{'bp':round(float(np.mean(v))*1e4,1),'t':round(tstat(v),2),'n':len(v)}
                                    for q,v in cond[s0].items() if len(v)>10}
json.dump(res,open(f'{OUT}/metrics_v27_hidden_ind.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
