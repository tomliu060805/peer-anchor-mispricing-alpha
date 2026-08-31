# -*- coding: utf-8 -*-
"""v21b: 日频120d网络 + 30min 120d网络 双网络混合."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
src=open(PROJ+'/src/alpha/38_30min_net.py').read()
head=src.split("SIGS={}")[0]
exec(head)
engine=src.split("NHOLD,BM=200,3.0")[1].split("res={}")[0]
exec("NHOLD,BM=200,3.0"+engine)
SIGS={}
SIGS['daily120']=make_sig(z8['rebuilds'],z8['nb_p'],z8['wv_p'])
SIGS['m30_w120']=make_sig(RB,n21['w120_n'],n21['w120_w'])
PA,ZA=SIGS['daily120']; PB,ZB=SIGS['m30_w120']
PMIX=np.full((T,N),np.nan,np.float32)
for t in sig5:
    a,b2=rank_xs(PA[t]),rank_xs(PB[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        PMIX[t]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
ZMIX=np.full((T,N),np.nan,np.float32)
for t in sig5:
    a,b2=rank_xs(ZA[t]),rank_xs(ZB[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        ZMIX[t]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
res={}
res['mix120_gap']=ev(run(PMIX,ZA),'mix120_gap')
res['mix120_gapzs']=ev(run(PMIX,ZMIX),'mix120_gapzs')
json.dump(res,open(f'{OUT}/metrics_v21b_mix120.json','w'),ensure_ascii=False,indent=1)
print('done')
