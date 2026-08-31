# -*- coding: utf-8 -*-
"""v17: 四项升级实验 (100核).
[3] 社区锚: 21日网络top20边 -> Leiden社区 -> 社区均值动量锚 gap_comm; IC对照+混合产品
[4] 离散度预警卖出: 持仓股邻居动量离散度截面pct>=0.9 提前卖
[6] 10日重建网络: IC/产品对照 + 重要事件窗(2015股灾/2016熔断/2018熊/2021抱团瓦解/2022-04/2024微盘)
[7] 持有期上限: 持有>=20交易日且排名掉出前200 强制卖
产品统一 v13 配置(N200/B3/触板过滤/收盘成交/双边20bp), 基线同码重跑保证可比.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np, pandas as pd
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
t_end=int(np.searchsorted(dnum,END,side='right'))

def build_one(args):
    t1,k20=args
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
    def topk(kk):
        part=np.argpartition(-C,kk,axis=1)[:,:kk]
        rows=np.arange(n)[:,None]; vals=C[rows,part]
        o=np.argsort(-vals,axis=1)
        return part[rows,o],vals[rows,o]
    nb5,wv5=topk(K)
    out5n=np.full((N,K),-1,np.int32); out5w=np.zeros((N,K),np.float32)
    out5n[idx]=idx[nb5]; out5w[idx]=wv5
    if k20:
        nb20,wv20=topk(20)
        o20n=np.full((N,20),-1,np.int32); o20w=np.zeros((N,20),np.float32)
        o20n[idx]=idx[nb20]; o20w[idx]=wv20
        return t1,out5n,out5w,o20n,o20w
    return t1,out5n,out5w,None,None

rb21=list(range(WLIM,T,21)); rb10=list(range(WLIM,T,10))
if not os.path.exists(f'{CACHE}/nets_v17.npz'):
    with ProcessPoolExecutor(50) as ex:
        o21=list(ex.map(build_one,[(t,True) for t in rb21],chunksize=1))
        o10=list(ex.map(build_one,[(t,False) for t in rb10],chunksize=1))
    o21.sort(key=lambda x:x[0]); o10.sort(key=lambda x:x[0])
    np.savez_compressed(f'{CACHE}/nets_v17.npz',
        rb21=np.array(rb21,np.int32),
        n21_5=np.stack([o[1] for o in o21]),w21_5=np.stack([o[2] for o in o21]),
        n21_20=np.stack([o[3] for o in o21]),w21_20=np.stack([o[4] for o in o21]),
        rb10=np.array(rb10,np.int32),
        n10_5=np.stack([o[1] for o in o10]),w10_5=np.stack([o[2] for o in o10]))
    print('nets built',flush=True)
nv=np.load(f'{CACHE}/nets_v17.npz')
RB21,N21,W21=nv['rb21'],nv['n21_5'],nv['w21_5']
N21_20,W21_20=nv['n21_20'],nv['w21_20']
RB10,N10,W10=nv['rb10'],nv['n10_5'],nv['w10_5']

# ---- Leiden 社区 (21日网格) ----
import igraph as ig, leidenalg as la
COMM=np.full((len(RB21),N),-1,np.int32)
for b in range(len(RB21)):
    nb=N21_20[b]; wv=np.maximum(W21_20[b],0)
    src=np.repeat(np.arange(N),20); dst=nb.ravel(); w=wv.ravel()
    m=(dst>=0)&(w>0.05)
    src,dst,w=src[m],dst[m],w[m]
    lo=np.minimum(src,dst); hi=np.maximum(src,dst)
    key=lo.astype(np.int64)*N+hi
    _,ui=np.unique(key,return_index=True)
    gg=ig.Graph(n=N,edges=list(zip(lo[ui].tolist(),hi[ui].tolist())))
    part=la.find_partition(gg,la.RBConfigurationVertexPartition,weights=w[ui].tolist(),
                           resolution_parameter=1.0,seed=42)
    lab=np.array(part.membership,np.int32)
    # 孤立点(无边)社区大小1, 标-1
    sizes=np.bincount(lab)
    small=sizes[lab]<6
    lab[small]=-1
    COMM[b]=lab
    if b%40==0: print(f'leiden {b}/{len(RB21)} ncomm>=6: {int((sizes>=6).sum())}',flush=True)
print('communities ready',flush=True)

sig_days=[int(x) for x in np.arange(int(RB21[0])+1,t_end-7,5)]
def gap_from(RB,NB,WV):
    PG=np.full((T,N),np.nan,np.float32)
    for t in sig_days:
        b=np.searchsorted(RB,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wgt[idx]
        m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
        PG[t,idx[ok]]=agg[ok]-m[idx[ok]]
    return PG
def dsp_from(RB,NB,WV):
    D=np.full((T,N),np.nan,np.float32)
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
        D[t,idx]=np.sqrt(np.maximum(var,0))
    return D
def zs_from(RB,NB):
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
    return ZS
PG21=gap_from(RB21,N21,W21); ZS21=zs_from(RB21,N21); DSP21=dsp_from(RB21,N21,W21)
PG10=gap_from(RB10,N10,W10); ZS10=zs_from(RB10,N10)
# 社区锚
PGC=np.full((T,N),np.nan,np.float32)
for t in sig_days:
    b=np.searchsorted(RB21,t,side='right')-1
    lab=COMM[b]; m=mom[t]
    ok=(lab>=0)&~np.isnan(m)
    if ok.sum()<100: continue
    nc=lab.max()+1
    s=np.bincount(lab[ok],weights=m[ok],minlength=nc)
    c=np.bincount(lab[ok],minlength=nc)
    anchor=np.full(N,np.nan,np.float32)
    li=lab[ok]
    anchor_ok=(s[li]-m[ok])/np.maximum(c[li]-1,1)
    a=np.full(N,np.nan,np.float32); a[ok]=anchor_ok
    PGC[t]=a-m
print('signals ready',flush=True)

def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
def seg_of(t):
    if dnum[t]>=np.datetime64(DEV[0]) and dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]>=np.datetime64(VAL[0]) and dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))
res={}
# ---- [3]/[6] IC 对照 ----
def ic_dom(S,name):
    d={'dev':[],'val':[]}
    for t in sig_days:
        s=seg_of(t)
        if s is None: continue
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(S[t])
        f=np.exp(logc[min(t+HOLD,T-1)]-logc[t])-1
        a=rank_xs(np.where(dom,S[t],np.nan)); y=rank_xs(np.where(dom,f,np.nan))
        m2=~np.isnan(a)&~np.isnan(y)
        if m2.sum()<300: continue
        aa,yy=a[m2],y[m2]
        aa=(aa-aa.mean())/(aa.std()+1e-12); yy=(yy-yy.mean())/(yy.std()+1e-12)
        d[s].append(float((aa*yy).mean()))
    r={s:{'IC':round(float(np.mean(v)),4),'ICIR':round(float(np.mean(v)/(np.std(v)+1e-12)),3),'t':round(tstat(v),2)} for s,v in d.items()}
    print(name,json.dumps(r),flush=True)
    return r
res['IC_gap21']=ic_dom(PG21,'IC_gap21')
res['IC_gap_comm']=ic_dom(PGC,'IC_gap_comm')
BLEND=np.full((T,N),np.nan,np.float32)
for t in sig_days:
    a,b2=rank_xs(PG21[t]),rank_xs(PGC[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        BLEND[t]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
res['IC_gap_blend']=ic_dom(BLEND,'IC_gap_blend')
res['IC_gap10']=ic_dom(PG10,'IC_gap10')

# ---- 产品引擎 ----
NHOLD,BM=200,3.0
def combo_of(PGx,ZSx,t):
    a,b2=rank_xs(PGx[t]),rank_xs(ZSx[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        return np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
def run_product(PGx,ZSx,extra_sig=None,dsp_sell=False,age_cap=False):
    holdings={}; age={}; recs=[]
    for si,t in enumerate(sig_days):
        if si+1>=len(sig_days): break
        s=combo_of(PGx,ZSx,t)
        if extra_sig is not None:
            e=rank_xs(extra_sig[t])
            stk=np.stack([s,e])
            with np.errstate(all='ignore'):
                s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PGx[t])
        s=np.where(dom,s,np.nan)
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        if dsp_sell:
            dv=DSP21[t]; dr=rank_xs(dv)
            m2=~np.isnan(dr)
            pct=np.full(N,np.nan)
            if m2.sum()>100:
                q=np.argsort(np.argsort(np.where(m2,dv,np.nan)[m2]))/max(m2.sum()-1,1)
                pct[m2]=q
        new_h=dict(holdings)
        for i2 in list(new_h):
            sell=(rank[i2]>=BM*NHOLD) or np.isnan(s[i2])
            if dsp_sell and pct[i2]==pct[i2] and pct[i2]>=0.9: sell=True
            if age_cap and age.get(i2,0)>=20 and rank[i2]>=NHOLD: sell=True
            if sell and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<50: holdings={}; age={}; continue
        wt={i2:1.0/len(new_h) for i2 in new_h}
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig_days[si+1]
        pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        # 基准: 全指收盘
        recs.append((t,dnum[t],pr,turn/2))
        age={i2:age.get(i2,0)+(t_next-t) if i2 in holdings else 0 for i2 in wt}
        holdings=wt
    return recs
qz=np.load(f'{CACHE}/quanzhi_ret.npy'); lgq=np.cumsum(np.log1p(qz))
def ev(recs,name):
    ts=np.array([r[0] for r in recs]); ds=np.array([r[1] for r in recs])
    pr=np.array([r[2] for r in recs]); tn=np.array([r[3] for r in recs])
    bq=np.array([np.exp(lgq[sig_days[list(sig_days).index(t)+1]]-lgq[t])-1 for t in ts])
    ex=(1+pr)/(1+bq)-1; ann=252/HOLD
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',('2015-01-01','2024-08-16'))]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),
          'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
          'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),
          'turn':round(float(tn[m].mean()),3)}
    # 事件窗
    wins={'2015股灾':('2015-06-15','2015-09-30'),'2016熔断':('2016-01-01','2016-01-31'),
          '2018熊市':('2018-01-01','2018-12-31'),'2021抱团瓦解':('2021-01-01','2021-03-31'),
          '2022-04疫情':('2022-03-01','2022-04-30'),'2024微盘':('2024-01-01','2024-02-29')}
    r['event']={k:round(float(np.prod(1+ex[(ds>=np.datetime64(a))&(ds<=np.datetime64(b))])-1),4) for k,(a,b) in wins.items()}
    print(name,json.dumps(r,ensure_ascii=False),flush=True)
    return r
res['P_base']=ev(run_product(PG21,ZS21),'P_base')
res['P_comm_blend']=ev(run_product(PG21,ZS21,extra_sig=PGC),'P_comm_blend')
res['P_dsp_sell']=ev(run_product(PG21,ZS21,dsp_sell=True),'P_dsp_sell')
res['P_net10']=ev(run_product(PG10,ZS10),'P_net10')
res['P_agecap']=ev(run_product(PG21,ZS21,age_cap=True),'P_agecap')
json.dump(res,open(f'{OUT}/metrics_v17_upgrades.json','w'),ensure_ascii=False,indent=1)
print('done')
