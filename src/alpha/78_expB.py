# -*- coding: utf-8 -*-
"""v46(实验B): 三项低成本改进
 B1 信息强度收缩: gap × (邻居平均相关/域内中位相关), 弱锚退回中性
 B2 状态陈旧性降权: L2特征窗内成交清淡的重复样本降权(用有效成交日数调整)
 B3 状态化条件概率: (自身状态×邻居状态)离散→历史条件概率P(上涨)作第四锚"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/76_traj_analysis.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8,WV8=z8['nb_p'],z8['wv_p']
# B1: 锚质量 = 邻居平均相关
ANCHQ=np.full((T,N),np.nan,np.float32)
for t in sig5:
    b=np.searchsorted(RB8,t,side='right')-1
    wv=np.maximum(WV8[b],0); nb=NB8[b]
    ok=(nb>=0)
    q=np.where(ok.sum(1)>0,(wv*ok).sum(1)/np.maximum(ok.sum(1),1),np.nan)
    ANCHQ[t]=q
# B3: 状态化条件概率 (自身3态 × 邻居3态 → 下期上涨概率), 滚动104周历史
STATEP=np.full((T,N),np.nan,np.float32)
hist={}   # (s_self,s_peer) -> list of fwd sign
sig_sorted=sorted(sig5)
for k,t in enumerate(sig_sorted):
    if t+5>=T: break
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<300: continue
    m=mom20[t]; pmv=PM_b[t]
    dm=np.nanmean(m[ix])
    s_self=np.where(m[ix]-dm>np.nanquantile(m[ix]-dm,2/3),1,np.where(m[ix]-dm<np.nanquantile(m[ix]-dm,1/3),-1,0))
    s_peer=np.where(pmv[ix]-dm>np.nanquantile(pmv[ix]-dm,2/3),1,np.where(pmv[ix]-dm<np.nanquantile(pmv[ix]-dm,1/3),-1,0))
    # 用已有历史估概率
    p=np.full(len(ix),np.nan,np.float32)
    for a_ in [-1,0,1]:
        for b_ in [-1,0,1]:
            key=(a_,b_)
            h=hist.get(key,[])
            if len(h)>=200:
                p[(s_self==a_)&(s_peer==b_)]=float(np.mean(h[-20000:]))
    STATEP[t,ix]=p
    # 更新历史(用t期的实际结果, 不参与t期预测)
    f=np.exp(logc[t+5,ix]-logc[t,ix])-1
    fu=np.nanmean(f)
    up=(f>fu).astype(np.float32)
    for a_ in [-1,0,1]:
        for b_ in [-1,0,1]:
            sel=(s_self==a_)&(s_peer==b_)
            if sel.sum()>0: hist.setdefault((a_,b_),[]).extend(up[sel].tolist())
print('B3 state prob ready',flush=True)
nzf=np.load(f'{CACHE}/nets_v41_flow.npz')
PG_dual=np.full((T,N),np.nan,np.float32)
RBx=nzf['dual_t']; NBd=nzf['dual_n']; WVd=nzf['dual_w']
for t in sig5:
    b=np.searchsorted(RBx,t,side='right')-1
    if b<0: continue
    nbr=NBd[b]; wgt=np.maximum(WVd[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    if len(idx)<100: continue
    nb=nbr[idx]; w0=wgt[idx]
    m=mom20[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=np.maximum(w.sum(1),1e-9)
    mu=(np.nan_to_num(pm)*w).sum(1)/sw
    ok=w.sum(1)>1e-9
    PG_dual[t,idx[ok]]=mu[ok]-m[idx[ok]]
def run9(shrink=False,stale=False,stateanchor=False,wb=0.4,wh=0.4,ws=0.2,nhold=200):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        g1=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        g2=PG_dual[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        if shrink:
            aq=ANCHQ[t]; med=np.nanmedian(aq)
            sc_=np.clip(aq/max(med,1e-6),0.3,1.7)
            g1=g1*sc_; g2=g2*sc_
        parts=[rank_xs(g1),rank_xs(ZS_b[t]),rank_xs(g2)]
        if stateanchor: parts.append(rank_xs(STATEP[t]))
        stk=np.stack(parts)
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        base=np.where(dom,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        fm=dict(zip(FN,tfeat(t)))
        bl=[rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]
        behav_r=np.nanmean(np.stack(bl),0)
        if stale:
            # 成交清淡(5日成交额低于域内20%分位)的行为分向中性收缩
            tm5=tjc[t][IDX['total_money']]-tjc[t-5][IDX['total_money']]
            thin=tm5<np.nanquantile(tm5[hard],0.2)
            behav_r=np.where(thin,0.5+0.5*(behav_r-0.5),behav_r)
        no_board=((hl5[t]<1)&(ll5[t]<1)).astype(np.float32)
        no_quad=(~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))).astype(np.float32)
        px_ok=(np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)
        struct_r=(no_board+no_quad+px_ok)/3.0
        with np.errstate(all='ignore'):
            comb=wb*rank01(base,hard)+wh*np.nan_to_num(behav_r,nan=0.5)+ws*struct_r
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
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    return recs
def ev2(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'turn':round(float(tn[m].mean()),3)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v25_base']=ev2(run9(),'v25_base')
res['B1_收缩']=ev2(run9(shrink=True),'B1_收缩')
res['B2_陈旧降权']=ev2(run9(stale=True),'B2_陈旧降权')
res['B3_状态锚']=ev2(run9(stateanchor=True),'B3_状态锚')
res['B1+B2']=ev2(run9(shrink=True,stale=True),'B1+B2')
res['B1+B2+B3']=ev2(run9(shrink=True,stale=True,stateanchor=True),'B1+B2+B3')
json.dump(res,open(f'{OUT}/metrics_v46_expB.json','w'),ensure_ascii=False,indent=1)
print('done')
