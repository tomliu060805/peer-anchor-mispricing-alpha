# -*- coding: utf-8 -*-
"""v13 定型(触板过滤)逐年超额+超额净值图, 对中证全指, 双边20bp."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
exec(open(PROJ+'/src/alpha/17_v13_condition.py').read().split('# ========== A)')[0])
CH=PROJ+'/charts/v13_final_nav'
os.makedirs(CH,exist_ok=True)
qz=np.load(f'{CACHE}/quanzhi_ret.npy'); lgq=np.cumsum(np.log1p(qz))
NHOLD,BM=200,3.0
holdings={}; recs=[]
for t in np.arange(t_start,t_end-6,5):
    s=combo_at(t); dm=dom_mask(t); s=np.where(dm,s,np.nan)
    can_buy=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
    can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
    order=np.argsort(-np.nan_to_num(s,nan=-1e9))
    rank=np.full(N,1<<30); rank[order]=np.arange(N)
    new_h=dict(holdings)
    for i2 in list(new_h):
        if (rank[i2]>=BM*NHOLD or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
    for i2 in order:
        if len(new_h)>=NHOLD: break
        if i2 in new_h or not can_buy[i2] or np.isnan(s[i2]): continue
        new_h[i2]=0.0
    if len(new_h)<50: holdings={}; continue
    wt={i2:1.0/len(new_h) for i2 in new_h}
    turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
    te2=min(t+HOLD,T-1)
    pr=sum(w*(np.exp(logc[te2,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
    bq=np.exp(lgq[te2]-lgq[t])-1
    recs.append((dnum[t],pr,bq)); holdings=wt
ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
ex=(1+pr)/(1+bq)-1
np.savez(f'{OUT}/port_v13_board.npz',dates=ds.astype('datetime64[D]').astype(str),port=pr,bench=bq,ex=ex)
years=sorted(set(str(d)[:4] for d in ds))
yr={y:float(np.prod(1+ex[[str(d)[:4]==y for d in ds]])-1) for y in years}
print(json.dumps({k:round(v,4) for k,v in yr.items()},ensure_ascii=False))
fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(ds,np.cumprod(1+ex),lw=1.7,color='#d62728',label='v13 对中证全指 超额净值(费后, 双边20bp)')
ax.plot(ds,np.cumprod(1+pr),lw=1.0,color='#1f77b4',alpha=.65,label='组合绝对净值')
ax.plot(ds,np.cumprod(1+bq),lw=1.0,color='#7f7f7f',ls='--',alpha=.65,label='中证全指(000985)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.text(np.datetime64('2023-02-15'),0.8,'val段起点',fontsize=9,color='gray')
ax.set_yscale('log')
ax.set_title('v13 定型: 活跃域联动锚 N200/周调/缓冲带600/触板过滤 vs 中证全指')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); v=[yr[y] for y in years]
ax.bar(x,v,color=['#d62728' if q>=0 else '#2ca02c' for q in v])
for i,q in enumerate(v): ax.text(i,q+(0.006 if q>=0 else -0.022),f'{q*100:.1f}',ha='center',fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.grid(alpha=.3,axis='y')
plt.tight_layout(); plt.savefig(f'{CH}/v13_excess_vs_csiall.png',dpi=130); print('chart saved')
