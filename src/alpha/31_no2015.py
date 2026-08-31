# -*- coding: utf-8 -*-
"""v19r: 剔除2015后的修正口径 (dev 2016-2022 / val 2023-2024.08 / full 2016-2024.08).
重跑 v13基线 与 v19定型, 出修正版净值图."""
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
src=open(PROJ+'/src/alpha/28_quad_filter.py').read()
head=src.split("def run(")[0]
exec(head)
DEV=('2016-01-01','2022-12-31'); FULL=('2016-01-01','2024-08-16')
CH=PROJ+'/charts/v19r_exclude_2015'
os.makedirs(CH,exist_ok=True)
NHOLD,BM=200,3.0
def run(quad_filter):
    holdings={}; recs=[]
    for si,t in enumerate(sig_days):
        if si+1>=len(sig_days): break
        a,b2=rank_xs(PG[t]),rank_xs(ZS[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG[t])
        s=np.where(dom,s,np.nan)
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        if quad_filter:
            dm=np.nanmean(np.where(dom,mom[t],np.nan))
            selq&=~((PM[t]-dm>=0)&(mom[t]-dm>=0))
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=BM*NHOLD) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<50: holdings={}; continue
        wt={i2:1.0/len(new_h) for i2 in new_h}
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig_days[si+1]
        pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    return recs
def ev(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/HOLD
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
          'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),
          'win':round(float((e>0).mean()),3),'avg_bp':round(float(e.mean()*1e4),1),
          'pf':round(float(e[e>0].sum()/max(-e[e<0].sum(),1e-9)),2),'turn':round(float(tn[m].mean()),3),'n':int(len(e))}
    print(name,json.dumps(r),flush=True)
    return r,ds,pr,bq,ex
res={}
res['v13_no2015'],_,_,_,_=ev(run(False),'v13_no2015')
res['v19_no2015'],ds,pr,bq,ex=ev(run(True),'v19_no2015')
m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
ds,pr,bq,ex=ds[m],pr[m],bq[m],ex[m]
np.savez(f'{OUT}/port_v19_no2015.npz',dates=ds.astype('datetime64[D]').astype(str),port=pr,bench=bq,ex=ex)
years=sorted(set(str(d)[:4] for d in ds))
yr={y:float(np.prod(1+ex[[str(x)[:4]==y for x in ds]])-1) for y in years}
res['yearly']={k:round(v,4) for k,v in yr.items()}
json.dump(res,open(f'{OUT}/metrics_v19r_no2015.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res['yearly'],ensure_ascii=False))
nav=np.cumprod(1+ex)
fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(ds,nav,lw=1.7,color='#d62728',label='v19 对中证全指 超额净值(费后, 双边20bp)')
ax.plot(ds,np.cumprod(1+pr),lw=1.0,color='#1f77b4',alpha=.65,label='组合绝对净值')
ax.plot(ds,np.cumprod(1+bq),lw=1.0,color='#7f7f7f',ls='--',alpha=.65,label='中证全指(000985)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.text(np.datetime64('2023-02-15'),nav.max()*0.5,'val段起点',fontsize=9,color='gray')
ax.set_yscale('log')
ax.set_title('v19 定型 (回测起点2016, 剔除2015): 活跃域联动锚 N200/周调/缓冲带600/触板过滤/剔同涨掉队')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); v=[yr[y] for y in years]
ax.bar(x,v,color=['#d62728' if q>=0 else '#2ca02c' for q in v])
for i,q in enumerate(v): ax.text(i,q+(0.005 if q>=0 else -0.018),f'{q*100:.1f}',ha='center',fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.grid(alpha=.3,axis='y')
plt.tight_layout(); plt.savefig(f'{CH}/v19_excess_vs_csiall_from2016.png',dpi=130); print('chart saved')
