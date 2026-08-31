# -*- coding: utf-8 -*-
"""v54: alpha(v27选股) × beta(噪声区间破位做空) 融合.
alpha: 日频净值(周内买入持有, 权重随个股收益漂移)
beta : 噪声区间空头腿(CNI2000/CSI1000, S2仓位每笔0.25单位), 日内入场次日开盘平, P&L记在平仓日
融合 : combined_d = alpha_d + h * beta_d   (h=对冲比例, beta本身已是净收益率口径)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/86_lean_v27.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
# ---------- 1. alpha 日频净值 ----------
def alpha_daily(mode='E3'):
    holdings={}; daily={}
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        v=np.maximum(vol20[t]*np.sqrt(20),1e-4)
        g1v=PG_b[t]; PMx=PM_b[t]; fm=dict(zip(FN,tfeat(t))); momq=mom20[t]
        domv=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        parts=[rank_xs(g1v/v),rank_xs(ZS_b[t]),rank_xs(PG_dual[t]/v),rank_xs(PG_te[t]/v),rank_xs(fgap(ROE,t))]
        stk=np.stack(parts)
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        base=np.where(domv,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&domv&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        behav_r=np.nanmean(np.stack([rank01(-fm['sweep_sell'],hard),rank01(-fm['osize_sell'],hard)]),0)
        sl=[(np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)]
        dr=DROE[t]; fin=hard&np.isfinite(dr)
        sl.append((np.nan_to_num(dr,nan=0)>=np.nanquantile(dr[fin],1/3)).astype(np.float32) if fin.sum()>200 else np.ones(N,np.float32))
        struct_r=np.mean(np.stack(sl),0)
        with np.errstate(all='ignore'):
            comb=0.4*rank01(base,hard)+0.4*np.nan_to_num(behav_r,nan=0.5)+0.2*struct_r
        comb=np.where(hard,comb,np.nan)
        order=np.argsort(-np.nan_to_num(comb,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=600) or np.isnan(comb[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=200: break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        w=np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h]); w=w/w.sum()
        wt=dict(zip(new_h,w))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        ids=np.array(list(wt.keys())); ws=np.array([wt[i] for i in ids])
        cur=ws.copy()
        for d in range(t+1,min(t_next,T)+1):
            r=np.nan_to_num(ret[d,ids],nan=0.0)
            pr_d=float((cur*r).sum()/max(cur.sum(),1e-12))
            if d==t+1: pr_d-=turn*COST      # 成本记在调仓后首日
            daily[d]=daily.get(d,0.0)+pr_d
            cur=cur*(1+r)
        holdings=wt
    ds=np.array(sorted(daily.keys()))
    return ds,np.array([daily[d] for d in ds])
# ---------- 2. beta 日频 P&L ----------
BP=PROJ+'/data/idx1m.parquet'
def beta_daily(code,K=2.5,NDAY=14,FEE=10e-4,DEC=(1000,1029,1129,1359),UNIT=0.25):
    df=pd.read_parquet(BP)
    gdf=df[df['code']==code]
    pc=gdf.pivot_table(index='date',columns='hm',values='close')
    po=gdf.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={Td:(pc[Td]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for Td in DEC}
    C,O=pc.values,po.values
    nxt_open=dopen.shift(-1).values
    idx_dates=pd.to_datetime(pc.index)
    pnl={}
    for di in range(len(pc.index)-1):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        for Td in DEC:
            s=sig[Td].iloc[di]
            if not np.isfinite(s): continue
            j=hp.get(Td)
            if j is None or j+1>=len(hms): continue
            lower=o0*(1-K*s)
            if not np.isfinite(C[di,j]) or C[di,j]>=lower: continue
            entry=O[di,j+1]; exitp=nxt_open[di]
            if not (np.isfinite(entry) and np.isfinite(exitp)): continue
            r=(entry/exitp-1)-FEE*2          # 做空: 入场价/出场价-1
            pnl[idx_dates[di+1]]=pnl.get(idx_dates[di+1],0.0)+UNIT*r   # 记在平仓日
    return pnl
def to_daily_arr(pnl,dnum_arr):
    out=np.zeros(len(dnum_arr))
    m={pd.Timestamp(d):i for i,d in enumerate(dnum_arr)}
    for k,v in pnl.items():
        i=m.get(pd.Timestamp(k))
        if i is not None: out[i]=v
    return out
print('building alpha daily...',flush=True)
ads,apr=alpha_daily()
adates=dnum[ads]
print(f'alpha daily days={len(ads)} {adates[0]}~{adates[-1]}',flush=True)
res={}
qz=np.load(f'{CACHE}/quanzhi_ret.npy')
bench_d=qz[ads]
def stats(r,name,bench=None):
    r=np.asarray(r); nav=np.cumprod(1+r); ann=252
    d={'AnnRet':round(float(nav[-1]**(ann/len(r))-1),4),'Vol':round(float(r.std()*np.sqrt(ann)),4),
       'Sharpe':round(float(r.mean()/(r.std()+1e-12)*np.sqrt(ann)),2),
       'MDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4)}
    if bench is not None:
        ex=(1+r)/(1+bench)-1; enav=np.cumprod(1+ex)
        d.update({'ExAnn':round(float(enav[-1]**(ann/len(ex))-1),4),
                  'ExIR':round(float(ex.mean()/(ex.std()+1e-12)*np.sqrt(ann)),2),
                  'ExMDD':round(float((1-enav/np.maximum.accumulate(enav)).max()),4)})
    print(name,json.dumps(d),flush=True)
    return d
res['alpha_only']=stats(apr,'alpha_only(v27日频)',bench_d)
for code,nm in [('CNI2000','CNI2000'),('CSI1000','CSI1000'),('CSI500','CSI500')]:
    bp=beta_daily(code)
    bd=to_daily_arr(bp,adates)
    res[f'beta_{nm}']=stats(bd,f'beta_only_{nm}')
    for h in [0.5,1.0,1.5,2.0]:
        res[f'fusion_{nm}_h{h}']=stats(apr+h*bd,f'fusion_{nm}_h{h}',bench_d)
json.dump(res,open(f'{OUT}/metrics_v54_fusion.json','w'),ensure_ascii=False,indent=1)
np.savez(f'{OUT}/fusion_daily.npz',dates=adates.astype(str),alpha=apr,bench=bench_d)
print('done')
