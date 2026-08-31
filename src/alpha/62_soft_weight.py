# -*- coding: utf-8 -*-
"""v32: 软加权 vs 硬过滤. 先按基础信号选池(500), 再用行为因子打分定权重.
行为分 behav = mean(rank(-sweep_sell), rank(-osize_sell), rank(nl5))  (越高=抛售越温和)
变体:
 A  v22硬过滤基线
 B1 池500全持, 权重∝behav秩
 B2 池500→按behav取前200, 权重∝base分
 B3 池500→combined=(base秩+behav秩)/2 取前200, 权重∝combined
 B4 池500→combined取前200, 权重∝behav秩
 B5 池300/700 敏感性(用B3规则)
基础过滤(全变体共有): 停牌/封板/触板/象限/低价; 逐笔与nl5仅在A中做硬剔除"""
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
    """在mask内转[0,1]秩, 其余NaN"""
    out=np.full(N,np.nan,np.float32)
    ix=np.where(mask&~np.isnan(x))[0]
    if len(ix)<10: return out
    r=np.argsort(np.argsort(x[ix])).astype(np.float32)/max(len(ix)-1,1)
    out[ix]=r
    return out
def run(mode,pool=500,nhold=200):
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
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
        selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
        nlv,_,_=feat5(t)
        fm=dict(zip(FN,tfeat(t)))
        if mode=='A':
            vv=nlv[dom&~np.isnan(nlv)]
            if len(vv)>200:
                lo=np.nanquantile(vv,1/3); selq&=~(np.nan_to_num(nlv,nan=9)<lo)
            for fn in ['sweep_sell','osize_sell']:
                v=fm[fn]; vv2=v[dom&~np.isnan(v)]
                if len(vv2)>200:
                    hi2=np.nanquantile(vv2,2/3); selq&=~(np.nan_to_num(v,nan=-9)>hi2)
            score=base
        else:
            cand=selq&dom&~np.isnan(base)
            if cand.sum()<50: holdings={}; continue
            bs=np.where(cand,base,np.nan)
            thr=np.nanquantile(bs[cand],max(0.0,1-pool/max(cand.sum(),1)))
            inpool=cand&(bs>=thr)
            bh=np.nanmean(np.stack([rank01(-fm['sweep_sell'],inpool),rank01(-fm['osize_sell'],inpool),
                                    rank01(nlv,inpool)]),0)
            br=rank01(base,inpool)
            comb=np.nanmean(np.stack([br,bh]),0)
            if mode=='B1':   score=np.where(inpool,bh,np.nan); nh=pool
            elif mode=='B2': score=np.where(inpool&(bh>=np.nanquantile(bh[inpool],1-nhold/max(inpool.sum(),1))),br,np.nan); nh=nhold
            elif mode=='B3': score=np.where(inpool,comb,np.nan); nh=nhold
            elif mode=='B4': score=np.where(inpool,comb,np.nan); nh=nhold
            selq=inpool
        nh=pool if mode=='B1' else (200 if mode!='A' else 200)
        if mode=='A': nh=200
        order=np.argsort(-np.nan_to_num(score,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=BM*nh) or np.isnan(score[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=nh: break
            if i2 in new_h or np.isnan(score[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<max(nh//5,30): holdings={}; continue
        if mode=='A':
            w=np.array([max(score[i2],0)+1.0 for i2 in new_h])
        elif mode=='B1':
            w=np.array([np.nan_to_num(bh[i2],nan=0.5)+0.3 for i2 in new_h])
        elif mode=='B2':
            w=np.array([np.nan_to_num(br[i2],nan=0.5)+0.3 for i2 in new_h])
        elif mode=='B3':
            w=np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h])
        else:
            w=np.array([np.nan_to_num(bh[i2],nan=0.5)+0.3 for i2 in new_h])
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
    r['nhold_med']=int(np.median(diag))
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['A_v22hard']=ev(run('A'),'A_v22hard')
res['B1_pool500_all']=ev(run('B1',500),'B1_pool500_all')
res['B2_behavpick']=ev(run('B2',500),'B2_behavpick')
res['B3_comb_combw']=ev(run('B3',500),'B3_comb_combw')
res['B4_comb_behavw']=ev(run('B4',500),'B4_comb_behavw')
res['B3_pool300']=ev(run('B3',300),'B3_pool300')
res['B3_pool700']=ev(run('B3',700),'B3_pool700')
json.dump(res,open(f'{OUT}/metrics_v32_softweight.json','w'),ensure_ascii=False,indent=1)
print('done')
