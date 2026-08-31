# -*- coding: utf-8 -*-
"""v49: 基本面纳入(严格PIT). 数据源 financial_indicator 逐日快照(自带pub_date/stat_date).
PIT三重保险: ①按交易日文件取当日快照 ②只用 pub_date<=t 的行 ③再滞后1日
特征:
  roe_lv  ROE水平
  droe    ROE环比变化(vs 上一个不同stat_date的快照)
  sue     标准化未预期盈利(inc_net_profit_yoy 的截面标准化)
  gpm     毛利率
  fresh   距pub_date天数(信息新鲜度)
用法: A)基本面gap锚(邻居基本面均值-自身, 基本面改善但没涨) B)恶化过滤/软分 C)直接选股因子对照"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, STOCK_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/78_expB.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
code_ix={c:i for i,c in enumerate(codes)}
FI=STOCK_ROOT+'/fundamental/financial_indicator'
FCOLS=['roe','inc_net_profit_year_on_year','gross_profit_margin','net_profit_margin']
def load_fi(i):
    ds=str(dates[i]); f=f'{FI}/{ds}.parquet'
    out=np.full((len(FCOLS)+2,N),np.nan,np.float32)
    if not os.path.exists(f): return i,out
    try:
        d=pd.read_parquet(f,columns=['code','pub_date','stat_date']+FCOLS)
    except Exception:
        return i,out
    # PIT保险②: pub_date <= t-1 (再滞后1日)
    pdte=pd.to_datetime(d['pub_date'],errors='coerce')
    cutoff=pd.Timestamp(ds)-pd.Timedelta(days=1)
    d=d[pdte<=cutoff]
    if len(d)==0: return i,out
    ix=np.array([code_ix.get(c,-1) for c in d['code']]); okm=ix>=0
    for j,c in enumerate(FCOLS):
        v=pd.to_numeric(d[c],errors='coerce').values.astype(np.float32)
        out[j,ix[okm]]=v[okm]
    # stat_date 序号(用于识别新报告) + 新鲜度
    sd=pd.to_datetime(d['stat_date'],errors='coerce')
    out[len(FCOLS),ix[okm]]=(sd.values.astype('datetime64[D]').astype(np.float32))[okm]
    pdv=pd.to_datetime(d['pub_date'],errors='coerce')
    fresh=(pd.Timestamp(ds)-pdv).dt.days.values.astype(np.float32)
    out[len(FCOLS)+1,ix[okm]]=fresh[okm]
    return i,out
if not os.path.exists(f'{CACHE}/fi_grid.npz'):
    with ProcessPoolExecutor(45) as ex:
        outs=list(ex.map(load_fi,range(T),chunksize=8))
    FIG=np.full((T,len(FCOLS)+2,N),np.nan,np.float32)
    for i,o in outs: FIG[i]=o
    np.savez_compressed(f'{CACHE}/fi_grid.npz',fi=FIG)
    print('fi grid cached',flush=True)
FIG=np.load(f'{CACHE}/fi_grid.npz')['fi']
ROE=FIG[:,0]; SUE_RAW=FIG[:,1]; GPM=FIG[:,2]; NPM=FIG[:,3]; STAT=FIG[:,4]; FRESH=FIG[:,5]
# droe: 与60日前快照比(若stat_date变化)
DROE=np.full((T,N),np.nan,np.float32)
DROE[60:]=np.where(STAT[60:]>STAT[:-60],ROE[60:]-ROE[:-60],np.nan)
print('coverage roe:',float(np.isfinite(ROE[2000]).mean()),' droe:',float(np.isfinite(DROE[2000]).mean()),flush=True)
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8,WV8=z8['nb_p'],z8['wv_p']
def fgap(F,t):
    b=np.searchsorted(RB8,t,side='right')-1
    nbr=NB8[b]; wgt=np.maximum(WV8[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    nb=nbr[idx]; w0=wgt[idx]
    m=F[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=np.maximum(w.sum(1),1e-9)
    mu=(np.nan_to_num(pm)*w).sum(1)/sw
    out=np.full(N,np.nan,np.float32)
    ok=(w.sum(1)>1e-9)&~np.isnan(m[idx])
    out[idx[ok]]=m[idx[ok]]-mu[ok]     # 自身基本面 - 邻居基本面 (自己更好=正)
    return out
def run11(mode='base',wb=0.4,wh=0.4,ws=0.2,nhold=200):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        g1=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        g2=PG_dual[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        parts=[rank_xs(g1),rank_xs(ZS_b[t]),rank_xs(g2)]
        if mode=='fgap_anchor':
            parts.append(rank_xs(fgap(ROE,t)))
        elif mode=='fgap_droe':
            parts.append(rank_xs(fgap(DROE,t)))
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
        if mode=='behav_droe': bl.append(rank01(DROE[t],hard))
        if mode=='behav_sue': bl.append(rank01(SUE_RAW[t],hard))
        behav_r=np.nanmean(np.stack(bl),0)
        no_board=((hl5[t]<1)&(ll5[t]<1)).astype(np.float32)
        no_quad=(~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))).astype(np.float32)
        px_ok=(np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32)
        sl=[no_board,no_quad,px_ok]
        if mode=='struct_droe':
            dr=DROE[t]; good=(np.nan_to_num(dr,nan=0)>=np.nanquantile(dr[hard&np.isfinite(dr)],1/3) if np.isfinite(dr[hard]).sum()>200 else np.ones(N,bool))
            sl.append(good.astype(np.float32))
        struct_r=np.mean(np.stack(sl),0)
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
res={}
res['v25_base']=ev2(run11('base'),'v25_base')
res['基本面gap锚_ROE']=ev2(run11('fgap_anchor'),'基本面gap锚_ROE')
res['基本面gap锚_dROE']=ev2(run11('fgap_droe'),'基本面gap锚_dROE')
res['行为块加dROE']=ev2(run11('behav_droe'),'行为块加dROE')
res['行为块加SUE']=ev2(run11('behav_sue'),'行为块加SUE')
res['结构块加dROE过滤']=ev2(run11('struct_droe'),'结构块加dROE过滤')
json.dump(res,open(f'{OUT}/metrics_v49_fund.json','w'),ensure_ascii=False,indent=1)
print('done')
