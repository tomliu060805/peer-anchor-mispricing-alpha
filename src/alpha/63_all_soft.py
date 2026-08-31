# -*- coding: utf-8 -*-
"""v33: 全软打分版. 硬过滤仅保留机制性不可买(停牌/当日封板/ST/无信号); 其余六项全部软分数.
分块:
 base_r  = 联动锚信号秩
 behav_r = mean(rank(-扫单卖), rank(-卖单规模), rank(大单净流入))
 struct_r= mean(未触板1/0, 非同涨掉队1/0, 价格>=2元1/0)  (三项判断类)
变体: 权重配比 与 是否用综合分定仓位
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/59_tick_modulate.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
BM=3.0
def rank01(x,mask):
    out=np.full(N,np.nan,np.float32)
    ix=np.where(mask&~np.isnan(x))[0]
    if len(ix)<10: return out
    r=np.argsort(np.argsort(x[ix])).astype(np.float32)/max(len(ix)-1,1)
    out[ix]=r
    return out
def run(wb=0.5,wh=0.3,ws=0.2,nhold=200,wmode='comb',hard_struct=False):
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
        # 硬过滤: 仅机制性不可买
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        nlv,_,_=feat5(t); fm=dict(zip(FN,tfeat(t)))
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
        selq=hard.copy()
        if hard_struct:
            selq&=(no_board>0.5)&(no_quad>0.5)&(px_ok>0.5)
        order=np.argsort(-np.nan_to_num(comb,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=BM*nhold) or np.isnan(comb[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=nhold: break
            if i2 in new_h or np.isnan(comb[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        if wmode=='comb': w=np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h])
        elif wmode=='behav': w=np.array([np.nan_to_num(behav_r[i2],nan=0.5)+0.3 for i2 in new_h])
        else: w=np.ones(len(new_h))
        w=w/w.sum(); wt=dict(zip(new_h,w))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=sum(ww*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,ww in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); diag.append(len(new_h)); holdings=wt
    return recs,diag
def ev(rd,name):
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
    return r
res={}
res['S_532']=ev(run(0.5,0.3,0.2),'S_532')
res['S_442']=ev(run(0.4,0.4,0.2),'S_442')
res['S_433']=ev(run(0.4,0.3,0.3),'S_433')
res['S_622']=ev(run(0.6,0.2,0.2),'S_622')
res['S_334']=ev(run(0.3,0.3,0.4),'S_334')
res['S_442_behavw']=ev(run(0.4,0.4,0.2,wmode='behav'),'S_442_behavw')
res['S_442_ew']=ev(run(0.4,0.4,0.2,wmode='ew'),'S_442_ew')
res['S_442_N300']=ev(run(0.4,0.4,0.2,nhold=300),'S_442_N300')
res['S_442_hardstruct']=ev(run(0.4,0.4,0.2,hard_struct=True),'S_442_hardstruct')
json.dump(res,open(f'{OUT}/metrics_v33_allsoft.json','w'),ensure_ascii=False,indent=1)
print('done')
