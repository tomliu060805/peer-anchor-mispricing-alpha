# -*- coding: utf-8 -*-
"""v20 定型(v19+打分加权+波动率标准化) 净值序列+图, 2016起口径."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("res={}")[0]
exec(head)
CH=PROJ+'/charts/v20_final_nav'
os.makedirs(CH,exist_ok=True)
recs,hold=run(wmode='score',volnorm=True)
ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
tn=np.array([r[3] for r in recs])
m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
ds,pr,bq,tn=ds[m],pr[m],bq[m],tn[m]
ex=(1+pr)/(1+bq)-1
np.savez(f'{OUT}/port_v20_final.npz',dates=ds.astype('datetime64[D]').astype(str),port=pr,bench=bq,ex=ex)
ann=252/hold; nav=np.cumprod(1+ex)
years=sorted(set(str(d)[:4] for d in ds))
yr={y:float(np.prod(1+ex[[str(x)[:4]==y for x in ds]])-1) for y in years}
stats={'ExAnn':round(float(nav[-1]**(ann/len(ex))-1),4),'IR':round(float(ex.mean()/ex.std()*np.sqrt(ann)),2),
 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'win':round(float((ex>0).mean()),3),
 'avg_bp':round(float(ex.mean()*1e4),1),'pf':round(float(ex[ex>0].sum()/-ex[ex<0].sum()),2),
 'turn':round(float(tn.mean()),3),'n':int(len(ex))}
print(json.dumps({'stats':stats,'yearly':{k:round(v,4) for k,v in yr.items()}},ensure_ascii=False))
fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(ds,nav,lw=1.7,color='#d62728',label='v20 对中证全指 超额净值(费后, 双边20bp)')
ax.plot(ds,np.cumprod(1+pr),lw=1.0,color='#1f77b4',alpha=.65,label='组合绝对净值')
ax.plot(ds,np.cumprod(1+bq),lw=1.0,color='#7f7f7f',ls='--',alpha=.65,label='中证全指(000985)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.text(np.datetime64('2023-02-15'),nav.max()*0.55,'val段起点',fontsize=9,color='gray')
ax.set_yscale('log')
ax.set_title('v20 定型(2016起): 活跃域联动锚 N200/周调/缓冲带600/触板+象限过滤/打分加权/波动率标准化')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); v=[yr[y] for y in years]
ax.bar(x,v,color=['#d62728' if q>=0 else '#2ca02c' for q in v])
for i,q in enumerate(v): ax.text(i,q+(0.005 if q>=0 else -0.018),f'{q*100:.1f}',ha='center',fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.grid(alpha=.3,axis='y')
plt.tight_layout(); plt.savefig(f'{CH}/v20_excess_vs_csiall.png',dpi=130); print('chart saved')
