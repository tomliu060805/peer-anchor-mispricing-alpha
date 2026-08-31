# -*- coding: utf-8 -*-
"""v21信号 周频IC/ICIR: 全活跃域 vs v21可买域(六重过滤后)."""
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
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))
acc={s:{'dom':[],'prod':[]} for s in ['dev','val']}
for t in sig5:
    s0=seg_of(t)
    if s0 is None: continue
    gap=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
    a,b2=rank_xs(gap),rank_xs(ZS_b[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        sc=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    dm=np.nanmean(np.where(dom,mom20[t],np.nan))
    selq=dom&(hl5[t]<1)&(ll5[t]<1)
    selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
    selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
    nlv,_,_=feat5(t)
    vv=nlv[dom&~np.isnan(nlv)]
    if len(vv)>200:
        lo=np.nanquantile(vv,1/3)
        selq&=~(np.nan_to_num(nlv,nan=9)<lo)
    f=np.exp(logc[min(t+5,T-1)]-logc[t])-1
    for key,mask in [('dom',dom),('prod',selq)]:
        cr=rank_xs(np.where(mask,sc,np.nan)); yr=rank_xs(np.where(mask,f,np.nan))
        m2=~np.isnan(cr)&~np.isnan(yr)
        if m2.sum()<200: continue
        c,y=cr[m2],yr[m2]
        c=(c-c.mean())/(c.std()+1e-12); y=(y-y.mean())/(y.std()+1e-12)
        acc[s0][key].append(float((c*y).mean()))
out={}
for s0 in ['dev','val']:
    out[s0]={k:{'IC':round(float(np.mean(v)),4),'ICIR':round(float(np.mean(v)/(np.std(v)+1e-12)),3),
                't':round(tstat(v),2),'n':len(v)} for k,v in acc[s0].items()}
json.dump(out,open(f'{OUT}/metrics_v26_ic_current.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
