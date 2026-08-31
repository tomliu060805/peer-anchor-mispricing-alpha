# -*- coding: utf-8 -*-
"""v8: 涨停共现网络的子域混淆检验 (100核并行).
1) 同覆盖对照: price:gap 只在 limitco 有信号的股票上评估
2) 信号相关性: rank corr(limitco_gap, price_gap)
3) 融合: 两信号秩平均(仅共同覆盖) vs 各自
4) 混合覆盖: limitco有边用limitco, 否则用price (全覆盖)
网络存盘 cache/nets_v8.npz"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np
from concurrent.futures import ProcessPoolExecutor

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,REBUILD,K,MOM,HOLD,COST,WLIM=120,21,5,20,5,0.00125,250
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
fwd=np.full((T,N),np.nan,np.float32)
fwd[:T-HOLD]=(np.exp(logc[HOLD:]-logc[:-HOLD])-1).astype(np.float32)
rebuilds=list(range(WLIM,T,REBUILD))

def build_one(t1):
    Rw=ret[t1-W:t1]
    valid=(~np.isnan(Rw)).sum(0)>=110
    valid&=~(st[t1-1]==1)
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
    def topk(C):
        part=np.argpartition(-C,K,axis=1)[:,:K]
        rows=np.arange(n)[:,None]; vals=C[rows,part]
        order=np.argsort(-vals,axis=1)
        return part[rows,order],vals[rows,order]
    nbp,wvp=topk(Cp)
    A=np.nan_to_num(at_hl[t1-WLIM:t1][:,idx],nan=0.0).astype(np.float32)
    Cl=A.T@A; np.fill_diagonal(Cl,-9)
    nbl,wvl=topk(Cl)
    wvl=np.where(wvl>=2,wvl,0)
    nb_p=np.full((N,K),-1,np.int32); wv_p=np.zeros((N,K),np.float32)
    nb_l=np.full((N,K),-1,np.int32); wv_l=np.zeros((N,K),np.float32)
    nb_p[idx]=idx[nbp]; wv_p[idx]=wvp
    nb_l[idx]=idx[nbl]; wv_l[idx]=wvl
    return t1,nb_p,wv_p,nb_l,wv_l

if not os.path.exists(f'{CACHE}/nets_v8.npz'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(build_one,rebuilds,chunksize=1))
    outs.sort(key=lambda x:x[0])
    NB_P=np.stack([o[1] for o in outs]); WV_P=np.stack([o[2] for o in outs])
    NB_L=np.stack([o[3] for o in outs]); WV_L=np.stack([o[4] for o in outs])
    np.savez_compressed(f'{CACHE}/nets_v8.npz',rebuilds=np.array(rebuilds,np.int32),
        nb_p=NB_P,wv_p=WV_P,nb_l=NB_L,wv_l=WV_L)
    print('nets built & saved',flush=True)
z=np.load(f'{CACHE}/nets_v8.npz')
rb=z['rebuilds']; NB_P,WV_P,NB_L,WV_L=z['nb_p'],z['wv_p'],z['nb_l'],z['wv_l']
sig_days=np.arange(rb[0]+1,T-1,HOLD)

def gap_from(NB,WV):
    S=np.full((len(sig_days),N),np.nan,np.float32)
    for si,t in enumerate(sig_days):
        b=np.searchsorted(rb,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        valid=(nbr[:,0]>=0)&(wgt.sum(1)>1e-9)
        idx=np.where(valid)[0]
        nb=nbr[idx]; w=wgt[idx]
        m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w*msk
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
        rows=idx[ok]
        S[si,rows]=agg[ok]-m[rows]
    return S
S_p=gap_from(NB_P,WV_P); S_l=gap_from(NB_L,WV_L)
def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
# 融合与混合
S_blend=np.full_like(S_p,np.nan); S_hybrid=S_p.copy()
sig_corr=[]
for si in range(len(sig_days)):
    a,b2=rank_xs(S_p[si]),rank_xs(S_l[si])
    both=~np.isnan(a)&~np.isnan(b2)
    if both.sum()>200: sig_corr.append(float(np.corrcoef(a[both],b2[both])[0,1]))
    stk=np.stack([a,b2]); S_blend[si]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
    hl=~np.isnan(S_l[si]); S_hybrid[si,hl]=S_l[si,hl]
print('signal rank-corr(price_gap, limitco_gap): mean=%.3f'%np.mean(sig_corr),flush=True)

def seg_mask(t,a,b): return (dnum[t]>=np.datetime64(a))and(dnum[t]<=np.datetime64(b))
def evaluate(S,name,mask_to=None):
    res={}; ann=252/HOLD
    for segn,(a,b) in [('dev',DEV),('val',VAL)]:
        ics=[];ls=[];te=[];prev=None;cov=[]
        for si,t in enumerate(sig_days):
            if not seg_mask(t,a,b): continue
            s=S[si].copy(); s[~tradable[t]]=np.nan
            if mask_to is not None: s[np.isnan(mask_to[si])]=np.nan
            f=fwd[t]
            m=~np.isnan(s)&~np.isnan(f)
            cov.append(int(m.sum()))
            if m.sum()<100: prev=None; continue
            sv,fv=s[m],f[m]; ids=np.where(m)[0]
            xr=np.argsort(np.argsort(sv)).astype(float); yr=np.argsort(np.argsort(fv)).astype(float)
            xr=(xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
            ics.append(float((xr*yr).mean()))
            q=np.argsort(np.argsort(sv))/len(sv)
            top=set(ids[q>=0.9])
            to=1-len(top&prev)/max(len(top),1) if prev else 1.0
            rt=fv[q>=0.9].mean(); rb2=fv[q<0.1].mean(); ru=fv.mean()
            ls.append(rt-rb2-to*2*COST*2); te.append(rt-ru-to*2*COST)
            prev=top
        ics,lsv,tev=map(np.array,(ics,ls,te))
        res[segn]={'IC':round(float(ics.mean()),4),'ICIR':round(float(ics.mean()/(ics.std()+1e-12)),3),
          'Sharpe':round(float(lsv.mean()/(lsv.std()+1e-12)*np.sqrt(ann)),2),
          'TopEx_ann':round(float(tev.mean()*ann),3),'cov':int(np.mean(cov))}
    print(name,json.dumps(res),flush=True)
    return res
res={}
res['price_full']=evaluate(S_p,'price_full')
res['limitco']=evaluate(S_l,'limitco')
res['price_on_limitco_cov']=evaluate(S_p,'price_on_limitco_cov',mask_to=S_l)
res['blend_common']=evaluate(S_blend,'blend_common',mask_to=S_l)
res['hybrid_fullcov']=evaluate(S_hybrid,'hybrid_fullcov')
res['signal_rank_corr']=round(float(np.mean(sig_corr)),3)
json.dump(res,open(f'{OUT}/metrics_v8_limitco.json','w'),ensure_ascii=False,indent=1)
print('done')
