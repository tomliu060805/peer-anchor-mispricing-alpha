# -*- coding: utf-8 -*-
"""v20: 剔除2015口径下重测全部历史判负候选. 基线=v19(触板+象限过滤). 100核."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM,COST=120,5,20,250,0.0010
DEV=('2016-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); FULL=('2016-01-01','2024-08-16')
END=np.datetime64('2024-08-16')
g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
money=g['money']; paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st_g=np.load(f'{CACHE}/st_grid.npz')['is_st']
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom20=np.full((T,N),np.nan,np.float32); mom20[20:]=(logc[20:]-logc[:-20]).astype(np.float32)
mom10=np.full((T,N),np.nan,np.float32); mom10[10:]=(logc[10:]-logc[:-10]).astype(np.float32)
r2=np.nan_to_num(ret,nan=0.0)**2; c2=np.cumsum(r2,0)
vol20=np.full((T,N),np.nan,np.float32); vol20[20:]=np.sqrt((c2[20:]-c2[:-20])/20).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st_g<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); llm=np.nan_to_num(at_ll,nan=0.0)
chl=np.cumsum(hl,0); cll=np.cumsum(llm,0)
lim250=np.zeros((T,N),np.float32); lim250[WLIM:]=chl[WLIM:]-chl[:-WLIM]
hl5=np.zeros((T,N),np.float32); hl5[5:]=chl[5:]-chl[:-5]
ll5=np.zeros((T,N),np.float32); ll5[5:]=cll[5:]-cll[:-5]
qz=np.load(f'{CACHE}/quanzhi_ret.npy'); lgq=np.cumsum(np.log1p(qz))
t_end=int(np.searchsorted(dnum,END,side='right'))
lm=np.log(np.where(money>0,money,np.nan))

def build_net(args):
    t1,kind=args
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
    sd=Xr.std(0)+1e-12; Zp=(Xr-Xr.mean(0))/sd
    Cp=(Zp.T@Zp/W).astype(np.float32); np.fill_diagonal(Cp,-9)
    def topk(C,kk=K):
        part=np.argpartition(-C,kk,axis=1)[:,:kk]
        rows=np.arange(n)[:,None]; vals=C[rows,part]
        o=np.argsort(-vals,axis=1)
        return part[rows,o],vals[rows,o]
    def pack(nb,wv,kk=K):
        on=np.full((N,kk),-1,np.int32); ow=np.zeros((N,kk),np.float32)
        on[idx]=idx[nb]; ow[idx]=wv
        return on,ow
    if kind=='price': return (t1,)+pack(*topk(Cp))
    if kind=='volume' or kind=='dual':
        V=lm[t1-W:t1][:,idx]
        V=np.where(np.isnan(V),np.nanmean(V,0,keepdims=True),V)
        V=np.nan_to_num(V,nan=0.0); V=V-V.mean(0)
        vm=V.mean(1,keepdims=True)
        bv=(V*vm).sum(0)/np.maximum((vm*vm).sum(),1e-12)
        Vr=V-vm@bv[None,:]
        for u in np.unique(gi):
            sel=gi==u
            if sel.sum()>1: Vr[:,sel]-=Vr[:,sel].mean(1,keepdims=True)
        sdv=Vr.std(0)+1e-12; Zv=(Vr-Vr.mean(0))/sdv
        Cv=(Zv.T@Zv/W).astype(np.float32); np.fill_diagonal(Cv,-9)
        if kind=='volume': return (t1,)+pack(*topk(Cv))
        rp=np.argsort(np.argsort(-Cp,axis=1),axis=1)
        rv=np.argsort(np.argsort(-Cv,axis=1),axis=1)
        Cd=-(rp+rv).astype(np.float32); np.fill_diagonal(Cd,-1e9)
        nb,_=topk(Cd)
        rows=np.arange(n)[:,None]
        return (t1,)+pack(nb,Cp[rows,nb])
    if kind=='leadlag':
        Cll=Cp.copy()
        for L in [1,2]:
            C1=(Zp[:-L].T@Zp[L:]/(W-L)).astype(np.float32)
            Cll=np.maximum(Cll,np.maximum(C1,C1.T))
        np.fill_diagonal(Cll,-9)
        return (t1,)+pack(*topk(Cll))
    if kind=='limitco':
        A=np.nan_to_num(hl[t1-WLIM:t1][:,idx],nan=0.0).astype(np.float32)
        Cl=A.T@A; np.fill_diagonal(Cl,-9)
        nb,wv=topk(Cl)
        wv=np.where(wv>=2,wv,0)
        return (t1,)+pack(nb,wv)
rb21=list(range(WLIM,T,21))
if not os.path.exists(f'{CACHE}/nets_v20_alt.npz'):
    jobs=[(t,k) for k in ['volume','dual','leadlag','limitco'] for t in rb21]
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(build_net,jobs,chunksize=1))
    store={}
    for k in ['volume','dual','leadlag','limitco']:
        os_=sorted([o for o,(kk) in zip(outs,[j[1] for j in jobs]) if kk==k],key=lambda x:x[0])
        store[f'{k}_n']=np.stack([o[1] for o in os_]); store[f'{k}_w']=np.stack([o[2] for o in os_])
    np.savez_compressed(f'{CACHE}/nets_v20_alt.npz',rb=np.array(rb21,np.int32),**store)
    print('alt nets built',flush=True)
alt=np.load(f'{CACHE}/nets_v20_alt.npz')
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8,WV8=z8['nb_p'],z8['wv_p']
nv=np.load(f'{CACHE}/nets_v17.npz')
RB17=nv['rb21']; N17_20,W17_20=nv['n21_20'],nv['w21_20']
RB10,N10,W10=nv['rb10'],nv['n10_5'],nv['w10_5']

# 弹性重建额外网络
r10s=np.zeros(T); r10s[10:]=np.exp(lgq[10:]-lgq[:-10])-1
qv=np.array([np.std(qz[max(0,i-19):i+1]) for i in range(T)])
qvm=np.array([np.median(qv[max(0,i-249):i+1]) for i in range(T)])
tc=(r10s<=-0.08)|(r10s>=0.10)|((qvm>0)&(qv/np.maximum(qvm,1e-9)>=2.0))
extra=[]; last=-99
for t in range(WLIM,t_end):
    if tc[t] and t-last>=10: extra.append(t); last=t
need=[t for t in extra if t not in set(int(x) for x in RB8)]
with ProcessPoolExecutor(30) as ex:
    exo=list(ex.map(build_net,[(t,'price') for t in need],chunksize=1))
allel={int(t):(n,w) for t,n,w in zip(RB8,NB8,WV8)}
for t1,nn,ww in exo: allel[int(t1)]=(nn,ww)
RBE=np.array(sorted(allel.keys()),np.int32)
NBE=np.stack([allel[int(t)][0] for t in RBE]); WVE=np.stack([allel[int(t)][1] for t in RBE])
print('elastic nets ready',flush=True)

sig5=[int(x) for x in np.arange(int(RB8[0])+1,t_end-7,5)]
sig10=[int(x) for x in np.arange(int(RB8[0])+1,t_end-12,10)]
def make_sig(RB,NB,WV,momM=mom20,days=sig5,do_zs=True):
    PG=np.full((T,N),np.nan,np.float32); PM=np.full((T,N),np.nan,np.float32); D=np.full((T,N),np.nan,np.float32)
    for t in days:
        b=np.searchsorted(RB,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wgt[idx]
        m=momM[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=np.maximum(w.sum(1),1e-9)
        mu=(np.nan_to_num(pm)*w).sum(1)/sw
        var=(np.nan_to_num((pm-mu[:,None])**2)*w).sum(1)/sw
        ok=w.sum(1)>1e-9
        PG[t,idx[ok]]=mu[ok]-m[idx[ok]]; PM[t,idx[ok]]=mu[ok]; D[t,idx[ok]]=np.sqrt(np.maximum(var[ok],0))
    ZS=np.full((T,N),np.nan,np.float32)
    if do_zs:
        for b in range(len(RB)):
            t1=int(RB[b]); t0=t1-W; t2=min(int(RB[b+1]) if b+1<len(RB) else T,T-1)
            sds=[t for t in days if t1<=t<t2]
            if not sds: continue
            nbr=NB[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
            base=logc[t0-1] if t0>0 else np.zeros(N)
            Pw=np.exp(logc[t0:t1]-base); nbc=np.where(nb>=0,nb,0)
            kk=nb.shape[1]
            sj=Pw[:,nbc.ravel()].reshape(W,len(idx),kk)
            s_tr=Pw[:,idx][:,:,None]-sj
            mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
            d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
            wsm=np.exp(-d)*(nb>=0); wsm/=np.maximum(wsm.sum(1,keepdims=True),1e-12)
            for t in sds:
                Pt=np.exp(logc[t]-base)
                zv=(Pt[idx][:,None]-Pt[nbc]-mu)/sd2
                ZS[t,idx]=-np.nansum(zv*wsm,1)
    return PG,PM,D,ZS
print('building signals...',flush=True)
S={}
S['price']=make_sig(RB8,NB8,WV8)
S['price_h10']=make_sig(RB8,NB8,WV8,days=sig10)
S['mom10']=make_sig(RB8,NB8,WV8,momM=mom10)
S['net10']=make_sig(RB10,N10,W10)
S['elastic']=make_sig(RBE,NBE,WVE)
S['K20']=make_sig(RB17,N17_20,W17_20,do_zs=False)
for k in ['volume','dual','leadlag','limitco']:
    S[k]=make_sig(alt['rb'],alt[f'{k}_n'],alt[f'{k}_w'])
# K3 / mutual (从K5截取)
NB3=NB8[:,:,:3].copy(); WV3=WV8[:,:,:3].copy()
S['K3']=make_sig(RB8,NB3,WV3)
NBm=NB8.copy(); WVm=WV8.copy()
for b in range(len(RB8)):
    nbf=NB8[b]
    for col in range(K):
        j=nbf[:,col]; ok=j>=0
        rows=np.where(ok)[0]
        mut=np.zeros(N,bool)
        mut[rows]=(nbf[j[rows]]==rows[:,None]).any(1)
        WVm[b][~mut,col]=0
S['mutual']=make_sig(RB8,NBm,WVm)
print('signals ready',flush=True)
import igraph as ig, leidenalg as la
PGC=np.full((T,N),np.nan,np.float32)
COMM=np.full((len(RB17),N),-1,np.int32)
for b in range(len(RB17)):
    nb=N17_20[b]; wv=np.maximum(W17_20[b],0)
    src=np.repeat(np.arange(N),20); dst=nb.ravel(); w=wv.ravel()
    m=(dst>=0)&(w>0.05)
    lo=np.minimum(src[m],dst[m]); hi=np.maximum(src[m],dst[m]); w2=w[m]
    key=lo.astype(np.int64)*N+hi
    _,ui=np.unique(key,return_index=True)
    gg=ig.Graph(n=N,edges=list(zip(lo[ui].tolist(),hi[ui].tolist())))
    part=la.find_partition(gg,la.RBConfigurationVertexPartition,weights=w2[ui].tolist(),resolution_parameter=1.0,seed=42)
    lab=np.array(part.membership,np.int32)
    sizes=np.bincount(lab); lab[sizes[lab]<6]=-1
    COMM[b]=lab
for t in sig5:
    b=np.searchsorted(RB17,t,side='right')-1
    lab=COMM[b]; m=mom20[t]
    ok=(lab>=0)&~np.isnan(m)
    if ok.sum()<100: continue
    nc=lab.max()+1
    su=np.bincount(lab[ok],weights=m[ok],minlength=nc); cn=np.bincount(lab[ok],minlength=nc)
    li=lab[ok]
    a=np.full(N,np.nan,np.float32); a[ok]=(su[li]-m[ok])/np.maximum(cn[li]-1,1)
    PGC[t]=a-m
print('community ready',flush=True)

def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
PG_b,PM_b,DSP_b,ZS_b=S['price']
def run(sigset='price',days=sig5,hold=5,nhold=200,bm=3.0,wmode='ew',
        bdisc=False,avoid_fri=False,blend=None,dsp_sell=None,age_cap=False,exit_accel=False,volnorm=False):
    PGx,PMx,Dx,ZSx=S[sigset]
    if sigset=='K20': ZSx=ZS_b
    holdings={}; age={}; recs=[]; dsp_hist={}
    for si,t in enumerate(days):
        if si+1>=len(days): break
        gap=PGx[t]
        if volnorm: gap=gap/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZSx[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        if blend is not None:
            e=rank_xs(blend[t])
            stk=np.stack([s,e])
            with np.errstate(all='ignore'):
                s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PGx[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))   # v19象限过滤(统一用price网络的PM)
        if bdisc:
            B=(PM_b[t]-dm)>-(mom20[t]-dm)
            s=np.where(B&~np.isnan(s),s-0.5,s)
        tt=t
        if avoid_fri and int(dnum[t].astype(object).weekday())==4:
            tt=t+1
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[tt]<0.5)&(at_ll[tt]<0.5)
        if dsp_sell=='xsec':
            dr=DSP_b[t]; m2=~np.isnan(dr)
            pct=np.full(N,np.nan)
            if m2.sum()>100:
                q=np.argsort(np.argsort(dr[m2]))/max(m2.sum()-1,1); pct[m2]=q
        new_h=dict(holdings)
        for i2 in list(new_h):
            sell=(rank[i2]>=bm*nhold) or np.isnan(s[i2])
            if dsp_sell=='xsec' and pct[i2]==pct[i2] and pct[i2]>=0.9: sell=True
            if dsp_sell=='spike':
                h=dsp_hist.get(i2,[])
                if len(h)>=6 and DSP_b[t,i2]==DSP_b[t,i2] and DSP_b[t,i2]>=2.0*np.median(h): sell=True
            if age_cap and age.get(i2,0)>=20 and rank[i2]>=nhold: sell=True
            if exit_accel and PM_b[t,i2]==PM_b[t,i2] and (PM_b[t,i2]-dm<0) and (mom20[t,i2]-dm>=0): sell=True
            if sell and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=nhold: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        if wmode=='ew': wt={i2:1.0/len(new_h) for i2 in new_h}
        else:
            sc=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc/=sc.sum()
            wt=dict(zip(new_h,sc))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=days[si+1]
        te2=tt+(t_next-t)
        te2=min(te2,T-1)
        pr=sum(w*(np.exp(logc[te2,i2]-logc[tt,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[te2]-lgq[tt])-1
        recs.append((dnum[t],pr,bq,turn/2))
        for i2 in new_h:
            dsp_hist.setdefault(i2,[])
            if DSP_b[t,i2]==DSP_b[t,i2]: dsp_hist[i2]=dsp_hist[i2][-11:]+[DSP_b[t,i2]]
        age={i2:(age.get(i2,0)+(t_next-t) if i2 in holdings else 0) for i2 in wt}
        holdings=wt
    return recs,hold
def ev(rh,name):
    recs,hold=rh
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/hold
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2)}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['BASE_v19']=ev(run(),'BASE_v19')
res['bdisc']=ev(run(bdisc=True),'bdisc')
res['avoid_fri']=ev(run(avoid_fri=True),'avoid_fri')
res['comm_blend']=ev(run(blend=PGC),'comm_blend')
res['K20_blend']=ev(run(blend=S['K20'][0]),'K20_blend')
res['K20_replace']=ev(run(sigset='K20'),'K20_replace')
res['K3']=ev(run(sigset='K3'),'K3')
res['mutual']=ev(run(sigset='mutual'),'mutual')
res['dsp_xsec']=ev(run(dsp_sell='xsec'),'dsp_xsec')
res['dsp_spike']=ev(run(dsp_sell='spike'),'dsp_spike')
res['net10']=ev(run(sigset='net10'),'net10')
res['elastic']=ev(run(sigset='elastic'),'elastic')
res['age_cap']=ev(run(age_cap=True),'age_cap')
res['exit_accel']=ev(run(exit_accel=True),'exit_accel')
res['N100']=ev(run(nhold=100),'N100')
res['N50']=ev(run(nhold=50),'N50')
res['B1']=ev(run(bm=1.0),'B1')
res['B2']=ev(run(bm=2.0),'B2')
res['H10']=ev(run(sigset='price_h10',days=sig10,hold=10),'H10')
res['score_w']=ev(run(wmode='score'),'score_w')
res['mom10']=ev(run(sigset='mom10'),'mom10')
res['volnorm']=ev(run(volnorm=True),'volnorm')
for k in ['volume','dual','leadlag','limitco']:
    res[f'net_{k}']=ev(run(sigset=k),f'net_{k}')
json.dump(res,open(f'{OUT}/metrics_v20_retest.json','w'),ensure_ascii=False,indent=1)
print('done')
