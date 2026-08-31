# -*- coding: utf-8 -*-
"""v16: A) 邻居动量加权方式对照(等权/corr/corr²) 域内IC
B) T+1开盘集合竞价成交版回测 vs 收盘成交版(同代码对照), 对全指(开盘口径)超额.
开盘涨停买不进/跌停卖不出: 以 open/pre_close >= +9.5%/-9.5% 近似(20cm股少量误伤, 保守方向)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, INDEX_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM,HOLD,COST=120,5,20,250,5,0.0010
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')
g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
opn,close=g['open'],g['close']
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); ll=np.nan_to_num(at_ll,nan=0.0)
chl=np.cumsum(hl,0); cll=np.cumsum(ll,0)
lim250=np.zeros((T,N),np.float32); lim250[WLIM:]=chl[WLIM:]-chl[:-WLIM]
hl5=np.zeros((T,N),np.float32); hl5[5:]=chl[5:]-chl[:-5]
ll5=np.zeros((T,N),np.float32); ll5[5:]=cll[5:]-cll[:-5]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
t_start=int(rb[0])+1; t_end=int(np.searchsorted(dnum,END,side='right'))
sig_days=list(np.arange(t_start,t_end-7,5))
# 复权开盘
with np.errstate(all='ignore'):
    af=np.exp(logc)/np.where(close>0,close,np.nan)
    aopen=opn*af
open_ret=np.where((opn>0)&~np.isnan(ret),opn/ (close/(1+np.nan_to_num(ret,nan=0)))-1,np.nan)  # open/pre_close-1
# 指数开盘
def rdq(i):
    f=f'{INDEX_ROOT}/price/price_daily/{dates[i]}.parquet'
    if not os.path.exists(f): return i,np.nan,np.nan
    d=pd.read_parquet(f,columns=['code','open','close'])
    r=d[d['code']=='000985.XSHG']
    if len(r)==0: return i,np.nan,np.nan
    return i,float(r['open'].iloc[0]),float(r['close'].iloc[0])
if not os.path.exists(f'{CACHE}/quanzhi_oc.npz'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(rdq,range(T),chunksize=20))
    qo=np.full(T,np.nan); qc=np.full(T,np.nan)
    for i,o,c in outs: qo[i]=o; qc[i]=c
    np.savez(f'{CACHE}/quanzhi_oc.npz',open=qo,close=qc)
qq=np.load(f'{CACHE}/quanzhi_oc.npz'); qo,qc=qq['open'],qq['close']

def peer_gap_w(wmode):
    PG=np.full((T,N),np.nan,np.float32)
    for t in sig_days:
        b=np.searchsorted(rb,t,side='right')-1
        nbr=NB_P[b]; rho=np.maximum(WV_P[b],0)
        if wmode=='eq': wraw=np.where(rho>0,1.0,0.0)
        elif wmode=='corr': wraw=rho
        else: wraw=rho**2
        idx=np.where((nbr[:,0]>=0)&(wraw.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wraw[idx]
        m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
        PG[t,idx[ok]]=agg[ok]-m[idx[ok]]
    return PG
def seg_of(t):
    if dnum[t]>=np.datetime64(DEV[0]) and dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]>=np.datetime64(VAL[0]) and dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))
res={}
# ---- A) 加权方式 IC ----
for wm in ['eq','corr','corr2']:
    PGw=peer_gap_w(wm)
    icd={'dev':[],'val':[]}
    for t in sig_days:
        s=seg_of(t)
        if s is None: continue
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PGw[t])
        f=np.exp(logc[min(t+HOLD,T-1)]-logc[t])-1
        a=rank_xs(np.where(dom,PGw[t],np.nan)); y=rank_xs(np.where(dom,f,np.nan))
        m2=~np.isnan(a)&~np.isnan(y)
        if m2.sum()<300: continue
        aa,yy=a[m2],y[m2]; aa=(aa-aa.mean())/(aa.std()+1e-12); yy=(yy-yy.mean())/(yy.std()+1e-12)
        icd[s].append(float((aa*yy).mean()))
    res[f'A_gapIC_{wm}']={s:{'IC':round(float(np.mean(v)),4),'ICIR':round(float(np.mean(v)/(np.std(v)+1e-12)),3)} for s,v in icd.items()}
    print(f'A {wm}',json.dumps(res[f'A_gapIC_{wm}']),flush=True)

# ---- B) T+1开盘 vs 收盘 成交 (v13配置: combo+触板过滤) ----
PG=peer_gap_w('corr')
ZS=np.full((T,N),np.nan,np.float32)
for b in range(len(rb)):
    t1=rb[b]; t0=t1-W; t2=min(rb[b+1] if b+1<len(rb) else T,T-1)
    sds=[t for t in sig_days if t1<=t<t2]
    if not sds: continue
    nbr=NB_P[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
    base=logc[t0-1] if t0>0 else np.zeros(N)
    Pw=np.exp(logc[t0:t1]-base); nbc=np.where(nb>=0,nb,0)
    sj=Pw[:,nbc.ravel()].reshape(W,len(idx),K)
    s_tr=Pw[:,idx][:,:,None]-sj
    mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
    d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
    wsm=np.exp(-d); wsm/=np.maximum(wsm.sum(1,keepdims=True),1e-12)
    for t in sds:
        Pt=np.exp(logc[t]-base)
        zv=(Pt[idx][:,None]-Pt[nbc]-mu)/sd2
        ZS[t,idx]=-np.nansum(zv*wsm,1)
def combo_at(t):
    a,b2=rank_xs(PG[t]),rank_xs(ZS[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        return np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
NHOLD,BM=200,3.0
def run(mode):
    holdings={}; recs=[]
    for si,t in enumerate(sig_days):
        s=combo_at(t)
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG[t])
        s=np.where(dom,s,np.nan)
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        tx=t+1 if mode=='open' else t          # 成交日
        t_next=sig_days[si+1] if si+1<len(sig_days) else None
        if t_next is None: break
        tx_next=t_next+1 if mode=='open' else t_next
        if mode=='open':
            buy_ok=(paused[tx]<0.5)&(np.nan_to_num(open_ret[tx],nan=9)<0.095)
            sell_ok=(paused[tx]<0.5)&(np.nan_to_num(open_ret[tx],nan=-9)>-0.095)
            px=aopen
        else:
            buy_ok=np.ones(N,bool); sell_ok=np.ones(N,bool)
            px=np.exp(logc)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if (rank[i2]>=BM*NHOLD or np.isnan(s[i2])) and sell_ok[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2] or not buy_ok[i2]: continue
            if np.isnan(px[tx,i2]) or np.isnan(px[tx_next,i2]): continue
            new_h[i2]=0.0
        if len(new_h)<50: holdings={}; continue
        wt={i2:1.0/len(new_h) for i2 in new_h}
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        pr=0.0
        for i2,w in wt.items():
            p0,p1=px[tx,i2],px[tx_next,i2]
            pr+=w*((p1/p0-1) if (p0==p0 and p1==p1 and p0>0) else 0.0)
        pr-=turn*COST
        bq=(qo[tx_next]/qo[tx]-1) if mode=='open' else (qc[tx_next]/qc[tx]-1)
        recs.append((dnum[t],pr,bq,turn/2))
        holdings=wt
    return recs
def ev(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    tn=np.array([r[3] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/HOLD
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',('2015-01-01','2024-08-16'))]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),
          'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
          'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),
          'win':round(float((e>0).mean()),3),'avg_bp':round(float(e.mean()*1e4),1),'turn':round(float(tn[m].mean()),3)}
    years=sorted(set(str(d)[:4] for d in ds))
    r['yearly']={y:round(float(np.prod(1+ex[[str(d)[:4]==y for d in ds]])-1),4) for y in years}
    print(name,json.dumps(r),flush=True)
    return r
res['B_close_exec']=ev(run('close'),'B_close_exec')
res['B_open_exec']=ev(run('open'),'B_open_exec')
json.dump(res,open(f'{OUT}/metrics_v16_openexec.json','w'),ensure_ascii=False,indent=1)
print('done')
