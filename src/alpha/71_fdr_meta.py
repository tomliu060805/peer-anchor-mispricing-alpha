# -*- coding: utf-8 -*-
"""v40: 多重检验校正(方向5). 汇总本项目所有被测变体的IR→t值, 做Bonferroni/BH-FDR/
多重检验门槛, 给出'在做了N次尝试的背景下, 多大的t才算真发现'."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, glob, math
import numpy as np
OUT=PROJ+'/output/alpha'
YEARS=8.63  # 2016-01 ~ 2024-08-16
rows=[]
for f in sorted(glob.glob(f'{OUT}/metrics_v*.json')):
    try: d=json.load(open(f))
    except Exception: continue
    fn=os.path.basename(f)
    def walk(prefix,obj):
        if isinstance(obj,dict):
            if 'full' in obj and isinstance(obj['full'],dict) and 'IR' in obj['full']:
                ir=obj['full'].get('IR'); ex=obj['full'].get('ExAnn')
                if ir is not None and isinstance(ir,(int,float)) and not (isinstance(ir,float) and math.isnan(ir)):
                    rows.append((fn,prefix,float(ir),float(ex) if ex is not None else float('nan')))
                return
            for k,v in obj.items(): walk(f'{prefix}.{k}' if prefix else k, v)
    walk('',d)
seen=set(); uniq=[]
for fn,nm,ir,ex in rows:
    key=(fn,nm)
    if key in seen: continue
    seen.add(key); uniq.append((fn,nm,ir,ex))
ts=np.array([r[2]*math.sqrt(YEARS) for r in uniq])
N=len(uniq)
print(f'被测变体总数(有full IR记录) N={N}')
print(f't值分布: max={ts.max():.2f} p90={np.quantile(ts,0.9):.2f} median={np.median(ts):.2f} min={ts.min():.2f}')
# 单侧p值
from math import erf
def p_one(t): return 0.5*(1-erf(t/math.sqrt(2)))
ps=np.array([p_one(t) for t in ts])
alpha=0.05
bonf=alpha/N
def ppf(q):
    # Acklam 近似
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl_=0.02425
    if q<pl_:
        qq=math.sqrt(-2*math.log(q))
        return (((((c[0]*qq+c[1])*qq+c[2])*qq+c[3])*qq+c[4])*qq+c[5])/((((d[0]*qq+d[1])*qq+d[2])*qq+d[3])*qq+1)
    if q>1-pl_:
        qq=math.sqrt(-2*math.log(1-q))
        return -(((((c[0]*qq+c[1])*qq+c[2])*qq+c[3])*qq+c[4])*qq+c[5])/((((d[0]*qq+d[1])*qq+d[2])*qq+d[3])*qq+1)
    qq=q-0.5; r=qq*qq
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*qq/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
t_bonf=ppf(1-bonf)
order=np.argsort(ps)
bh_thresh=None
for i,idx in enumerate(order,1):
    if ps[idx]<=alpha*i/N: bh_thresh=ps[idx]
t_bh=ppf(1-bh_thresh) if bh_thresh else None
res={'N_variants':N,'t_max':round(float(ts.max()),2),
     'Bonferroni_alpha':round(float(bonf),6),'t_threshold_Bonferroni':round(float(t_bonf),2),
     't_threshold_BH_FDR5%':round(float(t_bh),2) if t_bh else None,
     'rule_of_thumb_t':3.0,
     'n_pass_bonf':int((ts>t_bonf).sum()),'n_pass_bh':int((ts>t_bh).sum()) if t_bh else None,
     'n_pass_rule_of_thumb3':int((ts>3.0).sum())}
top=sorted(uniq,key=lambda r:-r[2])[:15]
res['top15']=[{'file':r[0],'variant':r[1],'IR':r[2],'ExAnn':r[3],'t':round(r[2]*math.sqrt(YEARS),2)} for r in top]
json.dump(res,open(f'{OUT}/metrics_v40_fdr.json','w'),ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in res.items() if k!='top15'},ensure_ascii=False,indent=1))
print('--- top10 ---')
for r in res['top15'][:10]: print(f"  t={r['t']:5.2f} IR={r['IR']:.2f} Ex={r['ExAnn']:.3f}  {r['variant']}  [{r['file']}]")
