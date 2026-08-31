# -*- coding: utf-8 -*-
"""v53: 精简版(去时点强依赖低贡献项), 在 E2(14:50实时) 与 E3(T+1) 双口径下评估.
时点依赖度分级:
  强依赖当日数据(实时算不准): 触板过滤(当日触板未知), 象限(需当日动量), 价格gap(需当日收盘)
  中等: zspread(当日末点), 行为块(当日逐笔不全)
  无依赖: TE锚/双确认锚(21日重建网络), ROE gap, dROE(均已滞后), 低价(用t-1价可)
消融边际贡献(full): 卖单规模-5.4 / TE-1.2 / 缓冲带-2.1 / 活跃域-1.4 / 价格gap-0.6 /
  双确认-0.5 / 扫单-0.5 / zspread-0.4 / ROEgap-0.3 / 象限-0.3 / 低价-0.2 / dROE-0.2 / 触板-0.1
候选精简: 砍 触板(-0.1,强依赖) / 象限(-0.3,强依赖) / zspread(-0.4,中依赖) 三项
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/85_v26_exec.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
def run15(mode,drop_board=False,drop_quad=False,drop_zs=False,drop_pricegap=False):
    holdings={}; recs=[]
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
        parts=[]
        if not drop_pricegap: parts.append(rank_xs(g1v/v))
        if not drop_zs: parts.append(rank_xs(ZS_b[t]))
        parts+= [rank_xs(PG_dual[t]/v),rank_xs(PG_te[t]/v),rank_xs(fgap(ROE,t))]
        stk=np.stack(parts)
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        base=np.where(domv,base,np.nan)
        hard=(paused[t]<0.5)&seal_ok&domv&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(domv,momq,np.nan))
        behav_r=np.nanmean(np.stack([rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]),0)
        sl=[]
        if not drop_board: sl.append(board_ok.astype(np.float32))
        if not drop_quad: sl.append((~((PMx-dm>=0)&(momq-dm>=0))).astype(np.float32))
        sl.append((np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32))
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
            if ((rank[i2]>=600) or np.isnan(comb[i2])) and can_sell[i2] and sell_ok[i2]:
                sold.append(i2); del new_h[i2]
        bought=[]
        for i2 in order:
            if len(new_h)>=200: break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2] or not buy_ok[i2]: continue
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
    return recs
def ev4(recs,name):
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
cfgs=[('完整v26',{}),
      ('砍触板',{'drop_board':True}),
      ('砍触板+象限',{'drop_board':True,'drop_quad':True}),
      ('砍触板+象限+zspread',{'drop_board':True,'drop_quad':True,'drop_zs':True}),
      ('砍触板+象限+zs+价格gap',{'drop_board':True,'drop_quad':True,'drop_zs':True,'drop_pricegap':True})]
for nm,kw in cfgs:
    for mode in ['E2','E3']:
        res[f'{nm}_{mode}']=ev4(run15(mode,**kw),f'{nm}_{mode}')
json.dump(res,open(f'{OUT}/metrics_v53_lean.json','w'),ensure_ascii=False,indent=1)
print('done')
