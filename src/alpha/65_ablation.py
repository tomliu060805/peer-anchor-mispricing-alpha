# -*- coding: utf-8 -*-
"""v35: 三块消融 — 联动锚到底还有没有用?
 A0: base=0 (纯行为+结构)   H0: behav=0   S0: struct=0
 + 低base权重边界 0.1/0.15  + 纯base对照"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/64_v33_final.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
res={}
for nm,(wb,wh,ws) in {
  'base0_行为结构':(0.0,0.5,0.5),
  'behav0_锚结构':(0.5,0.0,0.5),
  'struct0_锚行为':(0.5,0.5,0.0),
  'b10':(0.1,0.45,0.45),
  'b15':(0.15,0.45,0.40),
  'b20':(0.20,0.40,0.40),
  'b30':(0.30,0.40,0.30),
  'pure_base':(1.0,0.0,0.0),
  'pure_behav':(0.0,1.0,0.0),
  'pure_struct':(0.0,0.0,1.0),
}.items():
    res[nm]=ev(run2(wb,wh,ws),nm)
json.dump(res,open(f'{OUT}/metrics_v35_ablation.json','w'),ensure_ascii=False,indent=1)
print('done')
