# -*- coding: utf-8 -*-
"""v41(方向2): 资金流网络. 用逐笔重构的主动净买流建网, 三种边定义:
 flow: 净流残差相关(剔市场+行业)
 cross: A的净流 vs B的收益 交叉相关(取对称最大)
 dual: 价格残差相关 与 净流残差相关 都在top-K候选内才连边
下游 v24 配置(行为块=sweep+osize), 只换锚块的网络来源."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/68_behav_decay.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
W,K=120,5
TBG=np.load(f'{CACHE}/tb_grid.npz')['tb']   # (T,6,N): total_money,asell_amt,asell_sweep,asell_orders,abuy_amt,abuy_sweep
with np.errstate(all='ignore'):
    NETFLOW=(TBG[:,4]-TBG[:,1])/np.maximum(TBG[:,0],1.0)   # 主动净买占成交额
NETFLOW=np.where(np.isfinite(NETFLOW)&(TBG[:,0]>0),NETFLOW,np.nan).astype(np.float32)
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=[int(x) for x in z8['rebuilds']]
def ind_arr_t(t1):
    dstr=str(dates[t1-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    return np.array([imap.get(c,'') for c in codes])
def build_flow(args):
    t1,kind=args
    Rw=ret[t1-W:t1]; Fw=NETFLOW[t1-W:t1]
    valid=((~np.isnan(Rw)).sum(0)>=110)&((~np.isnan(Fw)).sum(0)>=100)
    valid&=~(st_g[t1-1]==1)
    inds=ind_arr_t(t1); valid&=inds!=''
    idx=np.where(valid)[0]; n=len(idx)
    if n<200: return t1,None,None
    def resid(M):
        X=np.nan_to_num(M[:,idx],nan=0.0).astype(np.float64)
        mkt=X.mean(1,keepdims=True)
        beta=(X*mkt).sum(0)/np.maximum((mkt*mkt).sum(),1e-12)
        Xr=X-mkt@beta[None,:]
        gi=inds[idx]
        for u in np.unique(gi):
            sel=gi==u
            if sel.sum()>1: Xr[:,sel]-=Xr[:,sel].mean(1,keepdims=True)
        sd=Xr.std(0)+1e-12
        return (Xr-Xr.mean(0))/sd
    Zp=resid(Rw); Zf=resid(Fw)
    if kind=='flow':
        C=(Zf.T@Zf/W).astype(np.float32)
    elif kind=='cross':
        C1=(Zf.T@Zp/W).astype(np.float32)   # i的流 vs j的价
        C=np.maximum(C1,C1.T)
    else:  # dual
        Cp=(Zp.T@Zp/W).astype(np.float32); Cf=(Zf.T@Zf/W).astype(np.float32)
        np.fill_diagonal(Cp,-9); np.fill_diagonal(Cf,-9)
        rp=np.argsort(np.argsort(-Cp,axis=1),axis=1)
        rf=np.argsort(np.argsort(-Cf,axis=1),axis=1)
        C=np.where((rp<30)&(rf<30),Cp,-9).astype(np.float32)
    np.fill_diagonal(C,-9)
    part=np.argpartition(-C,K,axis=1)[:,:K]
    rows=np.arange(n)[:,None]; vals=C[rows,part]
    o=np.argsort(-vals,axis=1)
    nb,wv=part[rows,o],vals[rows,o]
    on=np.full((N,K),-1,np.int32); ow=np.zeros((N,K),np.float32)
    ok=vals[rows,o]>-8
    on[idx]=np.where(ok,idx[nb],-1); ow[idx]=np.maximum(np.where(ok,wv,0),0)
    return t1,on,ow
if not os.path.exists(f'{CACHE}/nets_v41_flow.npz'):
    store={}
    for kind in ['flow','cross','dual']:
        with ProcessPoolExecutor(20) as ex:
            outs=list(ex.map(build_flow,[(t,kind) for t in RB8],chunksize=1))
        outs=[o for o in outs if o[1] is not None]
        ts=np.array([o[0] for o in outs],np.int32)
        store[f'{kind}_t']=ts
        store[f'{kind}_n']=np.stack([o[1] for o in outs]); store[f'{kind}_w']=np.stack([o[2] for o in outs])
        print(f'{kind} net built {len(outs)}',flush=True)
    np.savez_compressed(f'{CACHE}/nets_v41_flow.npz',**store)
nz=np.load(f'{CACHE}/nets_v41_flow.npz')
def make_gap(RBx,NB,WV):
    PG=np.full((T,N),np.nan,np.float32)
    for t in sig5:
        b=np.searchsorted(RBx,t,side='right')-1
        if b<0: continue
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        if len(idx)<100: continue
        nb=nbr[idx]; w0=wgt[idx]
        m=mom20[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=np.maximum(w.sum(1),1e-9)
        mu=(np.nan_to_num(pm)*w).sum(1)/sw
        ok=w.sum(1)>1e-9
        PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
    return PG
GAPS={'price':PG_b}
for kind in ['flow','cross','dual']:
    GAPS[kind]=make_gap(nz[f'{kind}_t'],nz[f'{kind}_n'],nz[f'{kind}_w'])
print('gaps ready',flush=True)
def run5(gapkey,mix=None,wb=0.4,wh=0.4,ws=0.2,nhold=200):
    PGx=GAPS[gapkey]
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        g1=PGx[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        parts=[rank_xs(g1)]
        if gapkey=='price': parts.append(rank_xs(ZS_b[t]))
        if mix is not None:
            parts.append(rank_xs(GAPS[mix][t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)))
        stk=np.stack(parts)
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        base=np.where(dom,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        fm=dict(zip(FN,tfeat(t)))
        base_r=rank01(base,hard)
        behav_r=np.nanmean(np.stack([rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]),0)
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
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    return recs
def ev_full(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        if m.sum()<10: continue
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
                 'turn':round(float(tn[m].mean()),3)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['v24_价格锚']=ev_full(run5('price'),'v24_价格锚')
res['flow_资金流锚']=ev_full(run5('flow'),'flow_资金流锚')
res['cross_流价交叉']=ev_full(run5('cross'),'cross_流价交叉')
res['dual_双确认']=ev_full(run5('dual'),'dual_双确认')
res['price+flow混合']=ev_full(run5('price',mix='flow'),'price+flow混合')
res['price+dual混合']=ev_full(run5('price',mix='dual'),'price+dual混合')
json.dump(res,open(f'{OUT}/metrics_v41_flownet.json','w'),ensure_ascii=False,indent=1)
print('done')
