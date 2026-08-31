# -*- coding: utf-8 -*-
"""v8b: 域定义对照 — 自身涨停>=2次(250d) vs 涨停共现边>=2 的覆盖, price:gap 同信号."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, pickle, numpy as np
CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,HOLD,COST,WLIM=120,5,20,5,0.00125,250
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16')
g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
fwd=np.full((T,N),np.nan,np.float32)
fwd[:T-HOLD]=(np.exp(logc[HOLD:]-logc[:-HOLD])-1).astype(np.float32)
z=np.load(f'{CACHE}/nets_v8.npz')
rb=z['rebuilds']; NB_P,WV_P,NB_L=z['nb_p'],z['wv_p'],z['nb_l']
WV_L=z['wv_l']
sig_days=np.arange(rb[0]+1,T-1,HOLD)
hl=np.nan_to_num(at_hl,nan=0.0); chl=np.cumsum(hl,0)
own_lim=np.zeros((T,N),np.float32); own_lim[WLIM:]=chl[WLIM:]-chl[:-WLIM]

def gap_from(NB,WV):
    S=np.full((len(sig_days),N),np.nan,np.float32)
    for si,t in enumerate(sig_days):
        b=np.searchsorted(rb,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w=wgt[idx]
        m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w*msk
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
        S[si,idx[ok]]=agg[ok]-m[idx[ok]]
    return S
S_p=gap_from(NB_P,WV_P)
lim_cov=np.full((len(sig_days),N),False)
for si,t in enumerate(sig_days):
    b=np.searchsorted(rb,t,side='right')-1
    lim_cov[si]=(NB_L[b][:,0]>=0)&(WV_L[b].sum(1)>0)

def seg_mask(t,a,b): return (dnum[t]>=np.datetime64(a))and(dnum[t]<=np.datetime64(b))
def evaluate(S,name,dommask=None):
    res={}; ann=252/HOLD
    for segn,(a,b) in [('dev',DEV),('val',VAL)]:
        ics=[];ls=[];te=[];prev=None;cov=[]
        for si,t in enumerate(sig_days):
            if not seg_mask(t,a,b): continue
            s=S[si].copy(); s[~tradable[t]]=np.nan
            if dommask is not None: s[~dommask[si]]=np.nan
            f=fwd[t]; m=~np.isnan(s)&~np.isnan(f)
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
dom_own2=np.zeros_like(lim_cov)
dom_own1=np.zeros_like(lim_cov)
for si,t in enumerate(sig_days):
    dom_own2[si]=own_lim[t]>=2; dom_own1[si]=own_lim[t]>=1
res['price_dom_own_lim2']=evaluate(S_p,'price_dom_own_lim2',dom_own2)
res['price_dom_own_lim1']=evaluate(S_p,'price_dom_own_lim1',dom_own1)
res['price_dom_limcoedge']=evaluate(S_p,'price_dom_limcoedge',lim_cov)
res['price_dom_notlim']=evaluate(S_p,'price_dom_notlim',~lim_cov)
json.dump(res,open(f'{OUT}/metrics_v8b_domain.json','w'),ensure_ascii=False,indent=1)
print('done')
