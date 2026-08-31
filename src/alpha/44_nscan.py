# -*- coding: utf-8 -*-
"""v24b: v20完整配置下持仓股数扫描 N∈{100,150,200,300,400}."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("res={}")[0]
exec(head)
res={}
for nh in [100,150,200,300,400]:
    res[f'N{nh}']=(lambda rh: {seg:{'ExAnn':round(float(np.cumprod(1+np.array([( (1+r[1])/(1+r[2])-1) for r in rh[0] if (r[0]>=np.datetime64(a))&(r[0]<=np.datetime64(b))]))[-1]**((252/5)/max(len([1 for r in rh[0] if (r[0]>=np.datetime64(a))&(r[0]<=np.datetime64(b))]),1))-1),4)} for seg,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]})(None) if False else None
# 直接用run+简评
def ev(rh,name):
    recs,hold=rh
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/hold
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
for nh in [100,150,200,300,400]:
    res[f'N{nh}']=ev(run(wmode='score',volnorm=True,nhold=nh),f'N{nh}')
json.dump(res,open(f'{OUT}/metrics_v24b_nscan.json','w'),ensure_ascii=False,indent=1)
print('done')
