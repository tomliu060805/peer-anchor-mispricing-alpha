# -*- coding: utf-8 -*-
"""v29: 拥挤度检验. 若信号被机构广泛捕捉, 修复应逐年代前移(h1占h20比例上升)且总量衰减.
分三个子期(2016-18/2019-21/2022-24)比较多头腿 alpha 衰减曲线 h=1/2/3/5/10/20."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/48_combo_final.py').read()
head=src.split("res={}")[0]
exec(head)
close_raw=g['close']
def sub_of(t):
    d=dnum[t]
    if d<np.datetime64('2016-01-01'): return None
    if d<np.datetime64('2019-01-01'): return 'P1_2016-18'
    if d<np.datetime64('2022-01-01'): return 'P2_2019-21'
    if d<=np.datetime64('2024-08-16'): return 'P3_2022-24'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
HS=[1,2,3,5,10,20]
acc={p:{h:[] for h in HS} for p in ['P1_2016-18','P2_2019-21','P3_2022-24']}
for t in sig5:
    p=sub_of(t)
    if p is None or t+21>=T: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    gap=PG_b[t,ix]/np.maximum(vol20[t,ix]*np.sqrt(20),1e-4)
    a,b2=rank_xs(PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)),rank_xs(ZS_b[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        sc=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))[ix]
    thr=np.nanquantile(sc,0.9); top=ix[sc>=thr]
    for h in HS:
        u=np.nanmean(np.exp(logc[t+h,ix]-logc[t,ix])-1)
        v=np.nanmean(np.exp(logc[t+h,top]-logc[t,top])-1)
        acc[p][h].append(float(v-u))
out={}
for p,d in acc.items():
    out[p]={f'h{h}':{'bp':round(float(np.mean(v))*1e4,1),'t':round(tstat(v),2)} for h,v in d.items()}
    h1=np.mean(d[1]); h20=np.mean(d[20])
    out[p]['h1_share_of_h20']=round(float(h1/h20),3) if h20>0 else None
    out[p]['n']=len(d[1])
json.dump(out,open(f'{OUT}/metrics_v29_crowding.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
