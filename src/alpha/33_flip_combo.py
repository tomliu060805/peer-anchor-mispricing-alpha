# -*- coding: utf-8 -*-
"""v20b: 三个翻案候选(打分加权/波动率标准化/避周五)的组合验证 + 逐年分解."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("res={}")[0]
exec(head)
def ev2(rh,name):
    recs,hold=rh
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/hold
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
          'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'win':round(float((e>0).mean()),3)}
    m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
    years=sorted(set(str(d)[:4] for d in ds[m]))
    r['yearly']={y:round(float(np.prod(1+ex[m][[str(x)[:4]==y for x in ds[m]]])-1),4) for y in years}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['base']=ev2(run(),'base')
res['sw']=ev2(run(wmode='score'),'sw')
res['vn']=ev2(run(volnorm=True),'vn')
res['fri']=ev2(run(avoid_fri=True),'fri')
res['sw_vn']=ev2(run(wmode='score',volnorm=True),'sw_vn')
res['sw_fri']=ev2(run(wmode='score',avoid_fri=True),'sw_fri')
res['vn_fri']=ev2(run(volnorm=True,avoid_fri=True),'vn_fri')
res['all3']=ev2(run(wmode='score',volnorm=True,avoid_fri=True),'all3')
json.dump(res,open(f'{OUT}/metrics_v20b_flipcombo.json','w'),ensure_ascii=False,indent=1)
print('done')
