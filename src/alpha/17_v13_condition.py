# -*- coding: utf-8 -*-
"""v13: A) 三维条件: 一致性(邻居动量离散)×活跃度×gap 的IC/多头超额网格 + gap/disp变体
B) 产品变体对全指: 基线 / +B型打折 / +触板过滤 / +避周五 / 全合并 (N200/H5/B3, 双边20bp)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, INDEX_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM,HOLD=120,5,20,250,5
COST=0.0010   # 单边10bp, 双边20bp
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); ll=np.nan_to_num(at_ll,nan=0.0)
chl=np.cumsum(hl,0); cll=np.cumsum(ll,0)
own_lim=np.zeros((T,N),np.float32); own_lim[WLIM:]=chl[WLIM:]-chl[:-WLIM]
hl5=np.zeros((T,N),np.float32); hl5[5:]=chl[5:]-chl[:-5]
ll5=np.zeros((T,N),np.float32); ll5[5:]=cll[5:]-cll[:-5]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
t_start=int(rb[0])+1
t_end=int(np.searchsorted(dnum,END,side='right'))

# ---- 日频: gap / peer_mom / 邻居离散 / zspread ----
PG=np.full((T,N),np.nan,np.float32); PM=np.full((T,N),np.nan,np.float32)
DSP=np.full((T,N),np.nan,np.float32)
for t in range(t_start,min(t_end+10,T-1)):
    b=np.searchsorted(rb,t,side='right')-1
    nbr=NB_P[b]; wgt=np.maximum(WV_P[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    nb=nbr[idx]; w0=wgt[idx]
    m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=w.sum(1); ok=sw>1e-9
    mu=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
    var=(np.nan_to_num((pm-mu[:,None])**2)*w).sum(1)/np.maximum(sw,1e-9)
    PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
    PM[t,idx[ok]]=mu[ok]
    DSP[t,idx[ok]]=np.sqrt(np.maximum(var[ok],0))
ZS=np.full((T,N),np.nan,np.float32)
for b in range(len(rb)):
    t1=rb[b]; t0=t1-W
    t2=min(rb[b+1] if b+1<len(rb) else T, T-1)
    if t1>=t_end+10: break
    nbr=NB_P[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
    base=logc[t0-1] if t0>0 else np.zeros(N)
    Pw=np.exp(logc[t0:t1]-base)
    nbc=np.where(nb>=0,nb,0)
    sj=Pw[:,nbc.ravel()].reshape(W,len(idx),K)
    s_tr=Pw[:,idx][:,:,None]-sj
    mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
    d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
    wsm=np.exp(-d); wsm/=np.maximum(wsm.sum(1,keepdims=True),1e-12)
    for t in range(t1,t2):
        Pt=np.exp(logc[t]-base)
        zv=(Pt[idx][:,None]-Pt[nbc]-mu)/sd2
        ZS[t,idx]=-np.nansum(zv*wsm,1)
print('daily signals ready',flush=True)

def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
def combo_at(t):
    a,b2=rank_xs(PG[t]),rank_xs(ZS[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        return np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
def dom_mask(t): return (own_lim[t]>=2)&tradable[t]&~np.isnan(PG[t])
def seg_of(t):
    if dnum[t]>=np.datetime64(DEV[0]) and dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]>=np.datetime64(VAL[0]) and dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,dtype=float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))

res={}
sig_days=np.arange(t_start,t_end-6,5)
# ========== A) 三维条件 ==========
grid_ic={s:{f'c{c}a{a}':[] for c in range(3) for a in range(3)} for s in ['dev','val']}
grid_te={s:{f'c{c}a{a}':[] for c in range(3) for a in range(3)} for s in ['dev','val']}
ic_plain={s:[] for s in ['dev','val']}; ic_scaled={s:[] for s in ['dev','val']}
for t in sig_days:
    s=seg_of(t)
    if s is None: continue
    dm=dom_mask(t); ix=np.where(dm)[0]
    if len(ix)<400: continue
    gv=PG[t,ix]; dv=DSP[t,ix]; av=own_lim[t,ix]
    f=np.exp(logc[t+HOLD,ix]-logc[t,ix])-1
    def ic_of(x,y):
        xr=np.argsort(np.argsort(x)).astype(float); yr=np.argsort(np.argsort(y)).astype(float)
        xr=(xr-xr.mean())/(xr.std()+1e-12); yr=(yr-yr.mean())/(yr.std()+1e-12)
        return float((xr*yr).mean())
    ic_plain[s].append(ic_of(gv,f))
    ic_scaled[s].append(ic_of(gv/(dv+np.nanmedian(dv)*0.5),f))
    ce=np.quantile(dv,[1/3,2/3]); ae=np.quantile(av,[1/3,2/3])
    cq=np.digitize(dv,ce); aq=np.digitize(av,ae)
    fu=f.mean()
    for c in range(3):
        for a2 in range(3):
            m2=(cq==c)&(aq==a2)
            if m2.sum()<60: continue
            grid_ic[s][f'c{c}a{a2}'].append(ic_of(gv[m2],f[m2]))
            gq=gv[m2]>=np.quantile(gv[m2],0.8)
            grid_te[s][f'c{c}a{a2}'].append(float(f[m2][gq].mean()-fu))
for s in ['dev','val']:
    res[f'A_ic_plain_{s}']={'IC':round(float(np.mean(ic_plain[s])),4),'t':round(tstat(ic_plain[s]),2)}
    res[f'A_ic_dispscaled_{s}']={'IC':round(float(np.mean(ic_scaled[s])),4),'t':round(tstat(ic_scaled[s]),2)}
    res[f'A_grid_ic_{s}']={k:{'IC':round(float(np.mean(v)),4),'t':round(tstat(v),2)} for k,v in grid_ic[s].items() if len(v)>20}
    res[f'A_grid_top_ex_{s}']={k:{'bp':round(float(np.mean(v))*1e4,1),'t':round(tstat(v),2)} for k,v in grid_te[s].items() if len(v)>20}
print('A done',flush=True)
for k in [k for k in res if k.startswith('A_')]: print(k,json.dumps(res[k],ensure_ascii=False))

# ========== B) 产品变体 ==========
def rd985(i):
    f=f'{INDEX_ROOT}/price/price_daily/{dates[i]}.parquet'
    if not os.path.exists(f): return i,np.nan
    d=pd.read_parquet(f,columns=['code','close','pre_close'])
    r=d[d['code']=='000985.XSHG']
    if len(r)==0: return i,np.nan
    return i,float(r['close'].iloc[0])/float(r['pre_close'].iloc[0])-1
if not os.path.exists(f'{CACHE}/quanzhi_ret.npy'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(rd985,range(T),chunksize=20))
    qz=np.zeros(T)
    for i,v in outs:
        if v==v: qz[i]=v
    np.save(f'{CACHE}/quanzhi_ret.npy',qz)
qz=np.load(f'{CACHE}/quanzhi_ret.npy'); lgq=np.cumsum(np.log1p(qz))

NHOLD,BM=200,3.0
def run_variant(bdisc=False,board_filter=False,avoid_fri=False):
    holdings={}; recs=[]
    t=t_start
    grid=list(np.arange(t_start,t_end-6,5))
    for t in grid:
        if avoid_fri and int(dnum[t].astype(object).weekday())==4:
            t=t+1
            if t>=t_end-6: break
        s=combo_at(t)
        dm=dom_mask(t)
        s=np.where(dm,s,np.nan)
        if bdisc:
            dmean=np.nanmean(mom[t][dm])
            B=(PM[t]-dmean)>-(mom[t]-dmean)
            s=np.where(B&~np.isnan(s),s-0.5,s)
        can_buy=(paused[t]<0.5)&(at_hl[t]<0.5)
        if board_filter: can_buy&=(hl5[t]<1)&(ll5[t]<1)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if (rank[i2]>=BM*NHOLD or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or not can_buy[i2] or np.isnan(s[i2]): continue
            new_h[i2]=0.0
        if len(new_h)<50: holdings={}; continue
        wt={i2:1.0/len(new_h) for i2 in new_h}
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        te2=min(t+HOLD,T-1)
        pr=sum(w*(np.exp(logc[te2,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[te2]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2))
        holdings=wt
    return recs
def eval_v(recs,name):
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
          'win':round(float((e>0).mean()),3),'avg_bp':round(float(e.mean()*1e4),1),
          'pf':round(float(e[e>0].sum()/max(-e[e<0].sum(),1e-9)),2),'turn':round(float(tn[m].mean()),3),'n':int(len(e))}
    print(name,json.dumps(r),flush=True)
    return r
res['B_baseline']=eval_v(run_variant(),'B_baseline')
res['B_bdisc']=eval_v(run_variant(bdisc=True),'B_bdisc')
res['B_board']=eval_v(run_variant(board_filter=True),'B_board')
res['B_nofri']=eval_v(run_variant(avoid_fri=True),'B_nofri')
res['B_all']=eval_v(run_variant(bdisc=True,board_filter=True,avoid_fri=True),'B_all')
json.dump(res,open(f'{OUT}/metrics_v13_condition.json','w'),ensure_ascii=False,indent=1)
print('done')
