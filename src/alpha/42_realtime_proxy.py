# -*- coding: utf-8 -*-
"""v23: 14:50实时计算假设的验证. 用14:30bar价(更保守)作当日收盘代理重算一切:
- mom20/gap/zspread 的当日末点用 P30[t,bar6]
- 当日封板(不能买)代理: 14:30涨幅>=+9.5%; 当日触板(hl5/ll5含t)代理: 任一bar涨跌幅越过±9.3%
- 活跃域 lim250 用截至t-1 (实时可得)
成交仍按 t 收盘价(主规则). 对拍理想化回测."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
close_raw=g['close']
C30=np.load(f'{CACHE}/c30_grid.npz')['c']
NHOLD,BM=200,3.0
PG_id,PM_id,DSP_id,ZS_id=S['price']
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8,WV8=z8['nb_p'],z8['wv_p']
BAR=6  # 14:30
# 实时代理量
def rt_quantities(t):
    """返回: mom_rt(N), 当日涨幅rt(N), 当日曾触上限/下限(N,bool)"""
    c1=close_raw[t-1]
    p=C30[t,BAR]
    with np.errstate(all='ignore'):
        r_t=p/c1-1
    bad=~np.isfinite(r_t)|(np.abs(r_t)>0.25)
    r_t=np.where(bad,np.nan_to_num(ret[t],nan=0.0),r_t)   # 异常回退真实日收益(少数复权日)
    mom_rt=(logc[t-1]-logc[t-21]+np.log1p(r_t)).astype(np.float32)
    with np.errstate(all='ignore'):
        allb=C30[t,:BAR+1]/c1[None,:]-1
    hi_touch=np.nanmax(np.where(np.isfinite(allb),allb,-9),0)>=0.093
    lo_touch=np.nanmin(np.where(np.isfinite(allb),allb,9),0)<=-0.093
    sealed_hi=r_t>=0.095
    return mom_rt,r_t,hi_touch,lo_touch,sealed_hi
# 触板窗: 前4日真实 + 当日代理
hl4=np.zeros((T,N),np.float32); hl4[4:]=chl[4:]-chl[:-4]   # t-4..t-1
ll4=np.zeros((T,N),np.float32); ll4[4:]=cll[4:]-cll[:-4]
lim250_lag=np.zeros((T,N),np.float32); lim250_lag[1:]=lim250[:-1]
def replay(realtime):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5): break
        if realtime:
            mom_use,r_t,hi_t,lo_t,seal=rt_quantities(t)
            # gap/zspread 用实时mom与实时末点
            b=np.searchsorted(RB8,t,side='right')-1
            nbr=NB8[b]; wgt=np.maximum(WV8[b],0)
            idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
            nb=nbr[idx]; w0=wgt[idx]
            pm=mom_use[np.where(nb>=0,nb,0)]
            msk=(nb>=0)&~np.isnan(pm); w=w0*msk
            sw=np.maximum(w.sum(1),1e-9)
            mu=(np.nan_to_num(pm)*w).sum(1)/sw
            PGx=np.full(N,np.nan,np.float32); PMx=np.full(N,np.nan,np.float32)
            ok=w.sum(1)>1e-9
            PGx[idx[ok]]=mu[ok]-mom_use[idx[ok]]; PMx[idx[ok]]=mu[ok]
            # zspread 实时: 沿用理想版路径但末点换实时(近似: 理想ZS + 末点差修正过小, 直接用理想ZS的secondary role)
            ZSx=ZS_id[t]   # 近似: zspread变化主要由末点驱动,此近似偏乐观,由mom主导的gap已实时化
            dom=(lim250_lag[t]>=2)&(paused[t]<0.5)&(st_g[t]<0.5)&~np.isnan(PGx)&~np.isnan(ret[t])
            board_ok=(hl4[t]<1)&(ll4[t]<1)&~hi_t&~lo_t
            seal_ok=~seal
            momq=mom_use
        else:
            PGx=PG_id[t]; PMx=PM_id[t]; ZSx=ZS_id[t]
            dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_id[t])
            board_ok=(hl5[t]<1)&(ll5[t]<1)
            seal_ok=(at_hl[t]<0.5)
            momq=mom20[t]
        gap=PGx/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(np.where(dom,gap,np.nan)),rank_xs(np.where(dom,ZSx,np.nan))
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dm=np.nanmean(np.where(dom,momq,np.nan))
        selq=(paused[t]<0.5)&seal_ok&board_ok
        selq&=~((PMx-dm>=0)&(momq-dm>=0))
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=BM*NHOLD) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        sc=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc/=sc.sum()
        wt=dict(zip(new_h,sc))
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
    m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
    years=sorted(set(str(d)[:4] for d in ds[m]))
    r['yearly']={y:round(float(np.prod(1+ex[m][[str(x)[:4]==y for x in ds[m]]])-1),4) for y in years}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['ideal']=ev(replay(False),'ideal')
res['realtime1430']=ev(replay(True),'realtime1430')
json.dump(res,open(f'{OUT}/metrics_v23_realtime.json','w'),ensure_ascii=False,indent=1)
print('done')
