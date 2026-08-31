# -*- coding: utf-8 -*-
"""v18: [A]弹性重建(高门槛触发) [B1]离散度组合刹车 [B2]离散度自身尖峰卖出. 基线=v13."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM,HOLD,COST=120,5,20,250,5,0.0010
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')
g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st_g=np.load(f'{CACHE}/st_grid.npz')['is_st']
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st_g<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); ll=np.nan_to_num(at_ll,nan=0.0)
chl=np.cumsum(hl,0); cll=np.cumsum(ll,0)
lim250=np.zeros((T,N),np.float32); lim250[WLIM:]=chl[WLIM:]-chl[:-WLIM]
hl5=np.zeros((T,N),np.float32); hl5[5:]=chl[5:]-chl[:-5]
ll5=np.zeros((T,N),np.float32); ll5[5:]=cll[5:]-cll[:-5]
qz=np.load(f'{CACHE}/quanzhi_ret.npy'); lgq=np.cumsum(np.log1p(qz))
t_end=int(np.searchsorted(dnum,END,side='right'))

# ---- [A] 触发日 ----
r10=np.zeros(T); r10[10:]=np.exp(lgq[10:]-lgq[:-10])-1
qv=np.array([np.std(qz[max(0,i-19):i+1]) for i in range(T)])
qvmed=np.array([np.median(qv[max(0,i-249):i+1]) for i in range(T)])
trig_cond=(r10<=-0.08)|(r10>=0.10)|((qvmed>0)&(qv/np.maximum(qvmed,1e-9)>=2.0))
extra=[]; last=-99
for t in range(WLIM,t_end):
    if trig_cond[t] and t-last>=10:
        extra.append(t); last=t
print('extra rebuild triggers:',len(extra))
print([str(dnum[t]) for t in extra],flush=True)

def build_one(t1):
    Rw=ret[t1-W:t1]
    valid=(~np.isnan(Rw)).sum(0)>=110
    valid&=~(st_g[t1-1]==1)
    dstr=str(dates[t1-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    inds=np.array([imap.get(c,'') for c in codes])
    valid&=inds!=''
    idx=np.where(valid)[0]; n=len(idx)
    X=np.nan_to_num(Rw[:,idx],nan=0.0).astype(np.float64)
    mkt=X.mean(1,keepdims=True)
    beta=(X*mkt).sum(0)/np.maximum((mkt*mkt).sum(),1e-12)
    Xr=X-mkt@beta[None,:]
    gi=inds[idx]
    for u in np.unique(gi):
        sel=gi==u
        if sel.sum()>1: Xr[:,sel]-=Xr[:,sel].mean(1,keepdims=True)
    sd=Xr.std(0)+1e-12; Z=(Xr-Xr.mean(0))/sd
    C=(Z.T@Z/W).astype(np.float32); np.fill_diagonal(C,-9)
    part=np.argpartition(-C,K,axis=1)[:,:K]
    rows=np.arange(n)[:,None]; vals=C[rows,part]
    o=np.argsort(-vals,axis=1)
    nb5,wv5=part[rows,o],vals[rows,o]
    outn=np.full((N,K),-1,np.int32); outw=np.zeros((N,K),np.float32)
    outn[idx]=idx[nb5]; outw[idx]=wv5
    return t1,outn,outw

nv=np.load(f'{CACHE}/nets_v17.npz')
RB21,N21,W21=[list(nv['rb21']),list(nv['n21_5']),list(nv['w21_5'])]
need=[t for t in extra if t not in set(RB21)]
with ProcessPoolExecutor(30) as ex:
    outs=list(ex.map(build_one,need,chunksize=1))
allnets={int(t):(n,w) for t,n,w in zip(RB21,N21,W21)}
for t1,nn,ww in outs: allnets[int(t1)]=(nn,ww)
RBM=np.array(sorted(allnets.keys()),np.int32)
NM=np.stack([allnets[int(t)][0] for t in RBM]); WM=np.stack([allnets[int(t)][1] for t in RBM])
print('merged rebuilds:',len(RBM),flush=True)

sig_days=[int(x) for x in np.arange(int(RBM[0])+1,t_end-7,5)]
def gap_zs_dsp(RB,NB,WV):
    PG=np.full((T,N),np.nan,np.float32); D=np.full((T,N),np.nan,np.float32)
    for t in sig_days:
        b=np.searchsorted(RB,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wgt[idx]
        m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=np.maximum(w.sum(1),1e-9)
        mu=(np.nan_to_num(pm)*w).sum(1)/sw
        var=(np.nan_to_num((pm-mu[:,None])**2)*w).sum(1)/sw
        ok=w.sum(1)>1e-9
        PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
        D[t,idx[ok]]=np.sqrt(np.maximum(var[ok],0))
    ZS=np.full((T,N),np.nan,np.float32)
    for b in range(len(RB)):
        t1=int(RB[b]); t0=t1-W; t2=min(int(RB[b+1]) if b+1<len(RB) else T,T-1)
        sds=[t for t in sig_days if t1<=t<t2]
        if not sds: continue
        nbr=NB[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
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
    return PG,ZS,D
PG_b,ZS_b,DSP_b=gap_zs_dsp(np.array(RB21),np.stack(N21),np.stack(W21))
PG_m,ZS_m,_=gap_zs_dsp(RBM,NM,WM)
print('signals ready',flush=True)
def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
# 组合级离散度 z (域内中位数, 过去250日稳健z, 只用<=t)
agg=np.full(T,np.nan)
for t in sig_days:
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(DSP_b[t])
    if dom.sum()>300: agg[t]=np.nanmedian(DSP_b[t][dom])
agg_z=np.full(T,np.nan)
sd_list=[t for t in sig_days if agg[t]==agg[t]]
for i,t in enumerate(sd_list):
    hist=[agg[x] for x in sd_list[max(0,i-50):i]]
    if len(hist)>=20:
        med=np.median(hist); mad=np.median(np.abs(np.array(hist)-med))+1e-9
        agg_z[t]=(agg[t]-med)/(1.4826*mad)
NHOLD,BM=200,3.0
def run(PGx,ZSx,brake=False,spike_sell=False):
    holdings={}; recs=[]; dsp_hist={}
    for si,t in enumerate(sig_days):
        if si+1>=len(sig_days): break
        a,b2=rank_xs(PGx[t]),rank_xs(ZSx[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PGx[t])
        s=np.where(dom,s,np.nan)
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            sell=(rank[i2]>=BM*NHOLD) or np.isnan(s[i2])
            if spike_sell:
                h=dsp_hist.get(i2,[])
                if len(h)>=6 and DSP_b[t,i2]==DSP_b[t,i2] and DSP_b[t,i2]>=2.0*np.median(h): sell=True
            if sell and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<50: holdings={}; continue
        expo=1.0
        if brake and agg_z[t]==agg_z[t] and agg_z[t]>=2.0: expo=0.5
        wt={i2:expo/len(new_h) for i2 in new_h}
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig_days[si+1]
        pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2,expo))
        for i2 in new_h:
            dsp_hist.setdefault(i2,[])
            if DSP_b[t,i2]==DSP_b[t,i2]: dsp_hist[i2]=dsp_hist[i2][-11:]+[DSP_b[t,i2]]
        holdings=wt
    return recs
def ev(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    tn=np.array([r[3] for r in recs]); ep=np.array([r[4] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/HOLD
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',('2015-01-01','2024-08-16'))]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e); anav=np.cumprod(1+pr[m])
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
          'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),
          'AbsAnn':round(float(anav[-1]**(ann/len(e))-1),4),
          'AbsSharpe':round(float(pr[m].mean()/(pr[m].std()+1e-12)*np.sqrt(ann)),2),
          'AbsMDD':round(float((1-anav/np.maximum.accumulate(anav)).max()),4),
          'turn':round(float(tn[m].mean()),3),'expo_mean':round(float(ep[m].mean()),3)}
    wins={'2015股灾':('2015-06-15','2015-09-30'),'2018熊市':('2018-01-01','2018-12-31'),
          '2020Q1':('2020-01-01','2020-03-31'),'2021Q1':('2021-01-01','2021-03-31'),
          '2024微盘':('2024-01-01','2024-02-29')}
    r['event_abs']={k:round(float(np.prod(1+pr[(ds>=np.datetime64(a))&(ds<=np.datetime64(b))])-1),4) for k,(a,b) in wins.items()}
    r['event_ex']={k:round(float(np.prod(1+ex[(ds>=np.datetime64(a))&(ds<=np.datetime64(b))])-1),4) for k,(a,b) in wins.items()}
    print(name,json.dumps(r,ensure_ascii=False),flush=True)
    return r
res={'trigger_days':[str(dnum[t]) for t in extra]}
res['base_v13']=ev(run(PG_b,ZS_b),'base_v13')
res['elastic_net']=ev(run(PG_m,ZS_m),'elastic_net')
res['dsp_brake']=ev(run(PG_b,ZS_b,brake=True),'dsp_brake')
res['dsp_spike_sell']=ev(run(PG_b,ZS_b,spike_sell=True),'dsp_spike_sell')
json.dump(res,open(f'{OUT}/metrics_v18_guard_elastic.json','w'),ensure_ascii=False,indent=1)
print('done')
