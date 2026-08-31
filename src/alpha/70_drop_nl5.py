# -*- coding: utf-8 -*-
"""v39: 行为块去掉nl5(金额阈值型大单净流入)的全段对照."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/68_behav_decay.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
def ev_full(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'turn':round(float(tn[m].mean()),3)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v23_三分量']=ev_full(run4('all'),'v23_三分量')
res['v24_去nl5']=ev_full(run4('sweep_osize'),'v24_去nl5')
res['仅sweep']=ev_full(run4('sweep'),'仅sweep')
res['仅osize']=ev_full(run4('osize'),'仅osize')
json.dump(res,open(f'{OUT}/metrics_v39_dropnl5.json','w'),ensure_ascii=False,indent=1)
print('done')
