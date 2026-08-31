# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
z=np.load(PROJ+'/output/alpha/port_v21_final.npz')
ds=z['dates'].astype('datetime64[D]'); pr=z['port']; bq=z['bench']; ex=z['ex']
nav=np.cumprod(1+ex); anav=np.cumprod(1+pr); bnav=np.cumprod(1+bq)
years=sorted(set(str(d)[:4] for d in ds))
yr={y:float(np.prod(1+ex[[str(x)[:4]==y for x in ds]])-1) for y in years}
fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(ds,nav,lw=1.7,color='#d62728',label='v21 对中证全指 超额净值(费后, 双边20bp)')
ax.plot(ds,anav,lw=1.0,color='#1f77b4',alpha=.65,label='组合绝对净值')
ax.plot(ds,bnav,lw=1.0,color='#7f7f7f',ls='--',alpha=.65,label='中证全指(000985)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.text(np.datetime64('2023-02-15'),0.72,'val段起点',fontsize=9,color='gray')
ax.set_yscale('log')
ticks=[0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.2]
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_minor_locator(FixedLocator([]))
fmt=ScalarFormatter(); fmt.set_scientific(False)
ax.yaxis.set_major_formatter(fmt)
ax.set_ylabel('净值')
ax.set_title('v21 定型(2016起): 活跃域联动锚 + 六重买入过滤 + 打分加权/波动率标准化')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); v=[yr[y] for y in years]
ax.bar(x,v,color=['#d62728' if q>=0 else '#2ca02c' for q in v])
for i,q in enumerate(v): ax.text(i,q+(0.006 if q>=0 else -0.02),f'{q*100:.1f}',ha='center',fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.grid(alpha=.3,axis='y')
plt.tight_layout()
plt.savefig(PROJ+'/charts/v21_final_nav/v21_excess_vs_csiall.png',dpi=130)
print('chart fixed')
