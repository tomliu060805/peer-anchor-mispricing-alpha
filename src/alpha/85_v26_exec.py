# -*- coding: utf-8 -*-
"""v52: v26 执行侧完整重算 (五锚+新变量口径).
 E1 理想: 收盘数据算+收盘成交 (基线)
 E2 路径A: 14:50实时(14:30bar代理当日末点/封板/触板; 逐笔用滞后窗; ROE/dROE本就滞后) + 尾盘成交
 E3 路径B: 收盘后完整数据算 + T+1拆分成交(11:30卖/13:30买)
 E4 朴素T+1开盘全量成交(最差对照)
诊断: 各路径换手/持仓/未成交阻塞率
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/82_combo_v26.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
close_raw=g['close']
C30=np.load(f'{CACHE}/c30_grid.npz')['c']
BAR_SELL,BAR_BUY,BAR_RT=3,4,6
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8,WV8=z8['nb_p'],z8['wv_p']
hl4=np.zeros((T,N),np.float32); hl4[4:]=chl[4:]-chl[:-4]
ll4=np.zeros((T,N),np.float32); ll4[4:]=cll[4:]-cll[:-4]
lim_lag=np.zeros((T,N),np.float32); lim_lag[1:]=lim250[:-1]
def rt_state(t):
    c1=close_raw[t-1]; p=C30[t,BAR_RT]
    with np.errstate(all='ignore'): r_t=p/c1-1
    bad=~np.isfinite(r_t)|(np.abs(r_t)>0.25)
    r_t=np.where(bad,np.nan_to_num(ret[t],nan=0.0),r_t)
    mom_rt=(logc[t-1]-logc[t-21]+np.log1p(r_t)).astype(np.float32)
    with np.errstate(all='ignore'): allb=C30[t,:BAR_RT+1]/c1[None,:]-1
    hi=np.nanmax(np.where(np.isfinite(allb),allb,-9),0)>=0.093
    lo=np.nanmin(np.where(np.isfinite(allb),allb,9),0)<=-0.093
    return mom_rt,hi,lo,(r_t>=0.095)
def run14(mode):
    holdings={}; recs=[]; blocked_buy=0; blocked_sell=0; want_buy=0; want_sell=0
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        v=np.maximum(vol20[t]*np.sqrt(20),1e-4)
        if mode=='E2':
            mom_use,hi_t,lo_t,seal=rt_state(t)
            b=np.searchsorted(RB8,t,side='right')-1
            nbr=NB8[b]; wgt=np.maximum(WV8[b],0)
            idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
            nb=nbr[idx]; w0=wgt[idx]
            pm=mom_use[np.where(nb>=0,nb,0)]
            msk=(nb>=0)&~np.isnan(pm); w=w0*msk
            sw=np.maximum(w.sum(1),1e-9); mu=(np.nan_to_num(pm)*w).sum(1)/sw
            g1v=np.full(N,np.nan,np.float32); PMx=np.full(N,np.nan,np.float32)
            ok=w.sum(1)>1e-9
            g1v[idx[ok]]=mu[ok]-mom_use[idx[ok]]; PMx[idx[ok]]=mu[ok]
            fm=dict(zip(FN,tfeat(t,lag=1)))
            momq=mom_use; board_ok=(hl4[t]<1)&(ll4[t]<1)&~hi_t&~lo_t; seal_ok=~seal
            domv=(lim_lag[t]>=2)&(paused[t]<0.5)&(st_g[t]<0.5)&~np.isnan(ret[t])&~np.isnan(g1v)
        else:
            g1v=PG_b[t]; PMx=PM_b[t]; fm=dict(zip(FN,tfeat(t))); momq=mom20[t]
            board_ok=(hl5[t]<1)&(ll5[t]<1); seal_ok=(at_hl[t]<0.5)
            domv=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        parts=[rank_xs(g1v/v),rank_xs(ZS_b[t]),rank_xs(PG_dual[t]/v),rank_xs(PG_te[t]/v),rank_xs(fgap(ROE,t))]
        stk=np.stack(parts)
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        base=np.where(domv,base,np.nan)
        hard=(paused[t]<0.5)&seal_ok&domv&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(domv,momq,np.nan))
        behav_r=np.nanmean(np.stack([rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]),0)
        sl=[board_ok.astype(np.float32),(~((PMx-dm>=0)&(momq-dm>=0))).astype(np.float32),
            (np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)]
        dr=DROE[t]; fin=hard&np.isfinite(dr)
        sl.append((np.nan_to_num(dr,nan=0)>=np.nanquantile(dr[fin],1/3)).astype(np.float32) if fin.sum()>200 else np.ones(N,np.float32))
        struct_r=np.mean(np.stack(sl),0)
        with np.errstate(all='ignore'):
            comb=0.4*rank01(base,hard)+0.4*np.nan_to_num(behav_r,nan=0.5)+0.2*struct_r
        comb=np.where(hard,comb,np.nan)
        order=np.argsort(-np.nan_to_num(comb,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        buy_ok=np.ones(N,bool); sell_ok=np.ones(N,bool)
        if mode in ('E3','E4') and t+1<T:
            c0=close_raw[t]; o1=C30[t+1,0]
            with np.errstate(all='ignore'): ovr=o1/c0-1
            buy_ok=(paused[t+1]<0.5)&(np.nan_to_num(ovr,nan=9)<0.095)
            sell_ok=(paused[t+1]<0.5)&(np.nan_to_num(ovr,nan=-9)>-0.095)
        new_h=dict(holdings); sold=[]
        for i2 in list(new_h):
            if (rank[i2]>=600) or np.isnan(comb[i2]):
                want_sell+=1
                if can_sell[i2] and sell_ok[i2]: sold.append(i2); del new_h[i2]
                else: blocked_sell+=1
        bought=[]
        for i2 in order:
            if len(new_h)>=200: break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2]: continue
            want_buy+=1
            if not buy_ok[i2]: blocked_buy+=1; continue
            new_h[i2]=0.0; bought.append(i2)
        if len(new_h)<40: holdings={}; continue
        w=np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h]); w=w/w.sum()
        wt=dict(zip(new_h,w))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]; pr=0.0
        if mode in ('E1','E2'):
            pr=sum(ww*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,ww in wt.items())
        else:
            bs=set(bought); bar_b=BAR_BUY if mode=='E3' else 0; bar_s=BAR_SELL if mode=='E3' else 0
            for i2,ww in wt.items():
                if i2 in bs and t+1<T:
                    p_in=C30[t+1,bar_b,i2]; c1=close_raw[t+1,i2]; c0=close_raw[t,i2]
                    if p_in==p_in and c1==c1 and c0==c0 and c0>0 and abs(C30[t+1,0,i2]/c0-1)<=0.15:
                        r=(c1/p_in)*np.exp(logc[t_next,i2]-logc[t+1,i2])-1
                    else: r=np.exp(logc[t_next,i2]-logc[t,i2])-1
                else: r=np.exp(logc[t_next,i2]-logc[t,i2])-1
                pr+=ww*r
            for i2 in sold:
                c0=close_raw[t,i2]; p_out=C30[t+1,bar_s,i2] if t+1<T else np.nan
                w_old=holdings.get(i2,0)
                if p_out==p_out and c0==c0 and c0>0 and abs(C30[t+1,0,i2]/c0-1)<=0.15:
                    pr+=w_old*(p_out/c0-1)
        pr-=turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    print(f'  [{mode}] 买入阻塞率 {blocked_buy/max(want_buy,1):.3f} 卖出阻塞率 {blocked_sell/max(want_sell,1):.3f}',flush=True)
    return recs
res={}
for mode,nm in [('E1','E1_理想'),('E2','E2_路径A_14:50实时+尾盘'),('E3','E3_路径B_T+1拆分'),('E4','E4_朴素T+1开盘')]:
    res[nm]=ev3(run14(mode),nm)
json.dump(res,open(f'{OUT}/metrics_v52_exec.json','w'),ensure_ascii=False,indent=1)
print('done')
