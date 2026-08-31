# -*- coding: utf-8 -*-
"""v22 持仓数诊断: 每期可买候选数/实际持仓数/未满仓期数占比 + 定型净值图."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
src=open(PROJ+'/src/alpha/59_tick_modulate.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
def run_v22(track=False):
    holdings={}; recs=[]; diag=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        gap=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZS_b[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        n_dom=int(dom.sum())
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
        selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
        nlv,_,_=feat5(t)
        vv=nlv[dom&~np.isnan(nlv)]
        if len(vv)>200:
            lo=np.nanquantile(vv,1/3); selq&=~(np.nan_to_num(nlv,nan=9)<lo)
        fm=dict(zip(FN,tfeat(t)))
        for fn in ['sweep_sell','osize_sell']:
            v=fm[fn]; vv2=v[dom&~np.isnan(v)]
            if len(vv2)>200:
                hi2=np.nanquantile(vv2,2/3); selq&=~(np.nan_to_num(v,nan=-9)>hi2)
        n_cand=int((selq&dom&~np.isnan(s)).sum())
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=3.0*200) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        n_kept=len(new_h)
        for i2 in order:
            if len(new_h)>=200: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        sc2=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc2/=sc2.sum()
        wt=dict(zip(new_h,sc2))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2))
        diag.append((dnum[t],n_dom,n_cand,n_kept,len(new_h)))
        holdings=wt
    return recs,diag
recs,diag=run_v22()
ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
ds,pr,bq,tn=ds[m],pr[m],bq[m],tn[m]
dg=[d for d in diag if (d[0]>=np.datetime64(FULL[0]))&(d[0]<=np.datetime64(FULL[1]))]
nd=np.array([d[1] for d in dg]); nc=np.array([d[2] for d in dg]); nh=np.array([d[4] for d in dg])
print('域内可选:',int(np.median(nd)),' 过滤后候选中位:',int(np.median(nc)),' 候选<200的期数占比:',round(float((nc<200).mean()),3))
print('实际持仓 中位/最小/满仓率:',int(np.median(nh)),int(nh.min()),round(float((nh>=200).mean()),3))
ex=(1+pr)/(1+bq)-1
np.savez(f'{OUT}/port_v22_final.npz',dates=ds.astype('datetime64[D]').astype(str),port=pr,bench=bq,ex=ex,nhold=nh,ncand=nc)
ann=252/5; nav=np.cumprod(1+ex)
years=sorted(set(str(d)[:4] for d in ds))
yr={y:float(np.prod(1+ex[[str(x)[:4]==y for x in ds]])-1) for y in years}
stats={'ExAnn':round(float(nav[-1]**(ann/len(ex))-1),4),'IR':round(float(ex.mean()/ex.std()*np.sqrt(ann)),2),
 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'win':round(float((ex>0).mean()),3),
 'avg_bp':round(float(ex.mean()*1e4),1),'pf':round(float(ex[ex>0].sum()/-ex[ex<0].sum()),2),'turn':round(float(tn.mean()),3)}
print(json.dumps({'stats':stats,'yearly':{k:round(v,4) for k,v in yr.items()}},ensure_ascii=False))
fig,axes=plt.subplots(3,1,figsize=(13,11),gridspec_kw={'height_ratios':[3,2,1.4]})
ax=axes[0]
ax.plot(ds,nav,lw=1.7,color='#d62728',label='v22 对中证全指 超额净值(费后, 双边20bp)')
ax.plot(ds,np.cumprod(1+pr),lw=1.0,color='#1f77b4',alpha=.65,label='组合绝对净值')
ax.plot(ds,np.cumprod(1+bq),lw=1.0,color='#7f7f7f',ls='--',alpha=.65,label='中证全指(000985)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.set_yscale('log')
ticks=[0.6,0.8,1.0,1.5,2.0,2.5,3.0]
ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([]))
fmt=ScalarFormatter(); fmt.set_scientific(False); ax.yaxis.set_major_formatter(fmt)
ax.set_ylabel('净值')
ax.set_title('v22 定型: 活跃域联动锚 + 八重过滤(含逐笔扫单/卖单规模) + 打分加权')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); v=[yr[y] for y in years]
ax.bar(x,v,color=['#d62728' if q>=0 else '#2ca02c' for q in v])
for i,q in enumerate(v): ax.text(i,q+(0.008 if q>=0 else -0.025),f'{q*100:.1f}',ha='center',fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.grid(alpha=.3,axis='y')
ax=axes[2]
ax.plot(ds,nc,lw=1.0,color='#ff7f0e',label='过滤后可买候选数')
ax.plot(ds,nh,lw=1.2,color='#1f77b4',label='实际持仓数')
ax.axhline(200,color='k',ls='--',lw=.8)
ax.set_title('持仓数诊断'); ax.legend(fontsize=9); ax.grid(alpha=.3)
plt.tight_layout()
CH=PROJ+'/charts/v22_final_nav'
os.makedirs(CH,exist_ok=True)
plt.savefig(f'{CH}/v22_excess_vs_csiall.png',dpi=130); print('chart saved')
