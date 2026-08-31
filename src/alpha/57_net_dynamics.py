# -*- coding: utf-8 -*-
"""v30b: 网络动力学. 邻居换血率churn(相邻两次重建top5交集), 条件表+产品过滤+市场级churn序列."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/48_combo_final.py').read()
head=src.split("res={}")[0]
exec(head)
close_raw=g['close']
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8=z8['nb_p']
B=len(RB8)
CHURN=np.full((B,N),np.nan,np.float32)
for b in range(1,B):
    prev=NB8[b-1]; cur=NB8[b]
    okc=(cur[:,0]>=0)&(prev[:,0]>=0)
    ix=np.where(okc)[0]
    inter=(cur[ix][:,:,None]==prev[ix][:,None,:]).any(2).sum(1)
    CHURN[b,ix]=1-inter/5.0
def churn_at(t):
    b=np.searchsorted(RB8,t,side='right')-1
    return CHURN[b]
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
res={}
# 条件表: 多头腿按churn三分位
acc={s:{q:[] for q in range(3)} for s in ['dev','val']}
mkt_churn=[]
for t in sig5:
    s0=seg_of(t)
    ch=churn_at(t)
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    mkt_churn.append((str(dnum[t]),float(np.nanmean(ch[ix]))))
    if s0 is None: continue
    a,b2=rank_xs(PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)),rank_xs(ZS_b[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        sc=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))[ix]
    f=np.exp(logc[min(t+5,T-1),ix]-logc[t,ix])-1
    fu=np.nanmean(f)
    thr=np.nanquantile(sc,0.9); top=sc>=thr
    cv=ch[ix]
    m2=top&~np.isnan(cv)
    if m2.sum()<20: continue
    qe=np.nanquantile(cv[m2],[1/3,2/3])
    qq=np.digitize(cv[m2],qe)
    for q in range(3):
        mm=qq==q
        if mm.sum()>=5: acc[s0][q].append(float(np.nanmean(f[m2][mm])-fu))
for s0 in ['dev','val']:
    res[f'cond_{s0}']={f'q{q}(churn{"低中高"[q]})':{'bp':round(float(np.mean(v))*1e4,1),'t':round(tstat(v),2)}
                       for q,v in acc[s0].items() if len(v)>20}
    print(f'cond_{s0}',json.dumps(res[f'cond_{s0}'],ensure_ascii=False),flush=True)
np.savez(f'{OUT}/mkt_churn_series.npz',dates=np.array([x[0] for x in mkt_churn]),churn=np.array([x[1] for x in mkt_churn]))
# 产品: 剔除churn最高1/3
def run_ch(use):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5): break
        gap=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZS_b[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
        selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
        nlv,_,_=feat5(t)
        vv=nlv[dom&~np.isnan(nlv)]
        if len(vv)>200:
            lo=np.nanquantile(vv,1/3)
            selq&=~(np.nan_to_num(nlv,nan=9)<lo)
        if use:
            ch=churn_at(t)
            cc=ch[dom&~np.isnan(ch)]
            if len(cc)>200:
                hi=np.nanquantile(cc,2/3)
                selq&=~(np.nan_to_num(ch,nan=-9)>hi)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=3.0*200) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
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
        recs.append((dnum[t],pr,bq)); holdings=wt
    return recs
def ev(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2)}
    print(name,json.dumps(r),flush=True)
    return r
res['prod_v21']=ev(run_ch(False),'prod_v21')
res['prod_churnfilter']=ev(run_ch(True),'prod_churnfilter')
json.dump(res,open(f'{OUT}/metrics_v30b_dynamics.json','w'),ensure_ascii=False,indent=1)
print('done')
