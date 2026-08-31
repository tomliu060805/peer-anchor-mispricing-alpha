# -*- coding: utf-8 -*-
"""v34: v33全软版收尾. 权重网格补点(确认高原) + 滞后窗稳健性 + 持仓诊断 + 定型净值图."""
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
src=open(PROJ+'/src/alpha/63_all_soft.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
def run2(wb,wh,ws,nhold=200,lag=False,track=False):
    holdings={}; recs=[]; diag=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        gap=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZS_b[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        base=np.where(dom,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        L=1 if lag else 0
        nlv=feat5(t)[0] if not lag else (lambda s: ((s[0]-s[1])/np.maximum(s[4],1.0)))(
            [ (mfc[t-1]-mfc[t-6])[j] for j in range(5) ])
        fm=dict(zip(FN,tfeat(t,lag=L)))
        base_r=rank01(base,hard)
        behav_r=np.nanmean(np.stack([rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard),
                                     rank01(nlv,hard)]),0)
        no_board=((hl5[t]<1)&(ll5[t]<1)).astype(np.float32)
        no_quad=(~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))).astype(np.float32)
        px_ok=(np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)
        struct_r=(no_board+no_quad+px_ok)/3.0
        with np.errstate(all='ignore'):
            comb=wb*base_r+wh*np.nan_to_num(behav_r,nan=0.5)+ws*struct_r
        comb=np.where(hard,comb,np.nan)
        order=np.argsort(-np.nan_to_num(comb,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=3.0*nhold) or np.isnan(comb[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=nhold: break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        w=np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h])
        w=w/w.sum(); wt=dict(zip(new_h,w))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=sum(ww*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,ww in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); diag.append((len(new_h),float(max(w)),int(hard.sum())))
        holdings=wt
    return recs,diag
def ev(rd,name,ret_series=False):
    recs,diag=rd
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'turn':round(float(tn[m].mean()),3)}
    print(name,json.dumps(r),flush=True)
    if ret_series: return r,ds,pr,bq,ex,tn,diag
    return r
res={}
for wb,wh,ws in [(0.2,0.4,0.4),(0.3,0.4,0.3),(0.35,0.35,0.3),(0.4,0.4,0.2),(0.3,0.3,0.4),(0.25,0.45,0.3)]:
    res[f'g_{int(wb*100)}_{int(wh*100)}_{int(ws*100)}']=ev(run2(wb,wh,ws),f'g_{wb}_{wh}_{ws}')
res['best_LAG']=ev(run2(0.4,0.4,0.2,lag=True),'best_LAG')
res['best_N150']=ev(run2(0.4,0.4,0.2,nhold=150),'best_N150')
res['best_N250']=ev(run2(0.4,0.4,0.2,nhold=250),'best_N250')
r,ds,pr,bq,ex,tn,diag=ev(run2(0.4,0.4,0.2),'FINAL_442',ret_series=True)
m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
ds,pr,bq,ex,tn=ds[m],pr[m],bq[m],ex[m],tn[m]
dg=[d for i,d in enumerate(diag) if m[i]]
nh=np.array([d[0] for d in dg]); mx=np.array([d[1] for d in dg]); nhard=np.array([d[2] for d in dg])
print('持仓中位/最小:',int(np.median(nh)),int(nh.min()),' 单票最大权重中位:',round(float(np.median(mx)),4),
      ' 硬过滤后候选中位:',int(np.median(nhard)),flush=True)
np.savez(f'{OUT}/port_v23_final.npz',dates=ds.astype('datetime64[D]').astype(str),port=pr,bench=bq,ex=ex,nhold=nh)
ann=252/5; nav=np.cumprod(1+ex)
years=sorted(set(str(d)[:4] for d in ds))
yr={y:float(np.prod(1+ex[[str(x)[:4]==y for x in ds]])-1) for y in years}
stats={'ExAnn':round(float(nav[-1]**(ann/len(ex))-1),4),'IR':round(float(ex.mean()/ex.std()*np.sqrt(ann)),2),
 'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),'win':round(float((ex>0).mean()),3),
 'avg_bp':round(float(ex.mean()*1e4),1),'pf':round(float(ex[ex>0].sum()/-ex[ex<0].sum()),2),'turn':round(float(tn.mean()),3)}
res['FINAL_stats']=stats; res['FINAL_yearly']={k:round(v,4) for k,v in yr.items()}
print(json.dumps({'stats':stats,'yearly':res['FINAL_yearly']},ensure_ascii=False))
json.dump(res,open(f'{OUT}/metrics_v34_final.json','w'),ensure_ascii=False,indent=1)
fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(ds,nav,lw=1.7,color='#d62728',label='v23 对中证全指 超额净值(费后, 双边20bp)')
ax.plot(ds,np.cumprod(1+pr),lw=1.0,color='#1f77b4',alpha=.65,label='组合绝对净值')
ax.plot(ds,np.cumprod(1+bq),lw=1.0,color='#7f7f7f',ls='--',alpha=.65,label='中证全指(000985)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.set_yscale('log')
ticks=[0.6,0.8,1.0,1.5,2.0,2.5,3.0,3.5]
ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([]))
fmt=ScalarFormatter(); fmt.set_scientific(False); ax.yaxis.set_major_formatter(fmt)
ax.set_ylabel('净值')
ax.set_title('v23 定型: 全软打分(硬剔仅机制不可买) 联动锚40%+微观行为40%+结构判断20%')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); v=[yr[y] for y in years]
ax.bar(x,v,color=['#d62728' if q>=0 else '#2ca02c' for q in v])
for i,q in enumerate(v): ax.text(i,q+(0.008 if q>=0 else -0.028),f'{q*100:.1f}',ha='center',fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.grid(alpha=.3,axis='y')
plt.tight_layout()
CH=PROJ+'/charts/v23_final_nav'
os.makedirs(CH,exist_ok=True)
plt.savefig(f'{CH}/v23_excess_vs_csiall.png',dpi=130); print('chart saved')
