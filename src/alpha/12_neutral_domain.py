# -*- coding: utf-8 -*-
"""v9: Barra(GYCNE5)中性化验真 + 域加权组合实验. T-1取暴露.
A) 中性化: signal_rank ~ 10风格 + 申万一级行业哑变量 -> 残差信号 IC/组合 (全市场 & 活跃域)
B) 活跃度连续单调性: 按250日涨停次数分5档, 档内因子IC
C) 域加权组合: 基线EW / 域内选股 / 软加权(活跃1.0安静0.3) vs 全市场等权基准
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, BARRA_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,HOLD,COST,WLIM=120,5,20,5,0.00125,250
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16')
STYLES=['BETA','BTOP','EARNYILD','GROWTH','LEVERAGE','LIQUIDTY','MOMENTUM','RESVOL','SIZE','SIZENL']

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
hl=np.nan_to_num(at_hl,nan=0.0); chl=np.cumsum(hl,0)
own_lim=np.zeros((T,N),np.float32); own_lim[WLIM:]=chl[WLIM:]-chl[:-WLIM]

z=np.load(f'{CACHE}/nets_v8.npz')
rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
sig_days=np.arange(rb[0]+1,T-1,HOLD)
code_ix={c:i for i,c in enumerate(codes)}

# ---- 信号: gap20 + zspread -> combo ----
S_pg=np.full((len(sig_days),N),np.nan,np.float32)
for si,t in enumerate(sig_days):
    b=np.searchsorted(rb,t,side='right')-1
    nbr=NB_P[b]; wgt=np.maximum(WV_P[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    nb=nbr[idx]; w=wgt[idx]
    m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w*msk
    sw=w.sum(1); ok=sw>1e-9
    agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
    S_pg[si,idx[ok]]=agg[ok]-m[idx[ok]]
S_zs=np.full((len(sig_days),N),np.nan,np.float32)
for b in range(len(rb)):
    t1=rb[b]; t0=t1-W
    t2=rb[b+1] if b+1<len(rb) else T
    sds=[(si,t) for si,t in enumerate(sig_days) if t1<=t<t2]
    if not sds: continue
    nbr=NB_P[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
    base=logc[t0-1] if t0>0 else np.zeros(N)
    Pw=np.exp(logc[t0:t1]-base)
    si_p=Pw[:,idx]; nbc=np.where(nb>=0,nb,0)
    sj_p=Pw[:,nbc.ravel()].reshape(W,len(idx),K)
    s_tr=si_p[:,:,None]-sj_p
    mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
    d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
    wsm=np.exp(-d); wsm/=np.maximum(wsm.sum(1,keepdims=True),1e-12)
    for si,t in sds:
        Pt=np.exp(logc[t]-base)
        s_t=Pt[idx][:,None]-Pt[nbc]
        zv=(s_t-mu)/sd2
        S_zs[si,idx]=-np.nansum(zv*wsm,1)
def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
COMBO=np.full_like(S_pg,np.nan)
for si in range(len(sig_days)):
    a,b2=rank_xs(S_pg[si]),rank_xs(S_zs[si])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        COMBO[si]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
print('signals ready',flush=True)

# ---- Barra 暴露加载 (T-1) ----
def load_expo(si):
    t=sig_days[si]; dprev=str(dates[t-1])
    out=np.full((N,len(STYLES)),np.nan,np.float32)
    for j,sty in enumerate(STYLES):
        f=f'{BARRA_ROOT}/{sty}/{dprev}.parquet'
        if not os.path.exists(f): return si,out
        d=pd.read_parquet(f)
        ix=[code_ix.get(c,-1) for c in d['code']]
        v=d[sty].values
        for kk,i2 in enumerate(ix):
            if i2>=0: out[i2,j]=v[kk]
    return si,out
with ProcessPoolExecutor(50) as ex:
    res_e=list(ex.map(load_expo,range(len(sig_days)),chunksize=4))
EXPO=np.full((len(sig_days),N,len(STYLES)),np.nan,np.float32)
for si,e in res_e: EXPO[si]=e
print('exposures loaded',flush=True)

def industry_dummy(t):
    dstr=str(dates[t-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    inds=np.array([imap.get(c,'') for c in codes])
    uniq=[u for u in np.unique(inds) if u]
    D=np.zeros((N,len(uniq)),np.float32)
    for j,u in enumerate(uniq): D[inds==u,j]=1
    return D

def neutralize(S):
    R=np.full_like(S,np.nan)
    for si,t in enumerate(sig_days):
        s=rank_xs(S[si])
        E=EXPO[si]; D=industry_dummy(t)
        m=~np.isnan(s)&~np.isnan(E).any(1)&tradable[t]
        if m.sum()<300: continue
        X=np.column_stack([np.ones(m.sum()),E[m],D[m]])
        keep=X.std(0)>1e-9; keep[0]=True
        b,_ ,_ ,_=np.linalg.lstsq(X[:,keep],s[m],rcond=None)
        R[si,m]=s[m]-X[:,keep]@b
    return R
COMBO_N=neutralize(COMBO)
print('neutralized',flush=True)

def seg_mask(t,a,b): return (dnum[t]>=np.datetime64(a))and(dnum[t]<=np.datetime64(b))
def evaluate(S,name,dom=None,weights=None):
    res={}; ann=252/HOLD
    for segn,(a,b) in [('dev',DEV),('val',VAL)]:
        ics=[];ls=[];te=[];prev=None;cov=[]
        for si,t in enumerate(sig_days):
            if not seg_mask(t,a,b): continue
            s=S[si].copy(); s[~tradable[t]]=np.nan
            if dom is not None: s[~dom[si]]=np.nan
            f=fwd[t]; m=~np.isnan(s)&~np.isnan(f)
            cov.append(int(m.sum()))
            if m.sum()<100: prev=None; continue
            sv,fv=s[m],f[m]; ids=np.where(m)[0]
            xr=np.argsort(np.argsort(sv)).astype(float); yr=np.argsort(np.argsort(fv)).astype(float)
            xr=(xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
            ics.append(float((xr*yr).mean()))
            q=np.argsort(np.argsort(sv))/len(sv)
            sel=q>=0.9
            top=set(ids[sel])
            to=1-len(top&prev)/max(len(top),1) if prev else 1.0
            if weights is not None:
                wv=weights[si][ids[sel]] if weights[si] is not None else np.ones(sel.sum())
                wv=np.maximum(wv,1e-9); wv=wv/wv.sum()
                rt=float((fv[sel]*wv).sum())
            else:
                rt=fv[sel].mean()
            rb2=fv[q<0.1].mean(); ru=fv.mean()
            ls.append(rt-rb2-to*2*COST*2); te.append(rt-ru-to*2*COST)
            prev=top
        ics,lsv,tev=map(np.array,(ics,ls,te))
        r=tev
        res[segn]={'IC':round(float(ics.mean()),4),'ICIR':round(float(ics.mean()/(ics.std()+1e-12)),3),
          'IC_t':round(float(ics.mean()/(ics.std()+1e-12)*np.sqrt(len(ics))),2),
          'Sharpe_LS':round(float(lsv.mean()/(lsv.std()+1e-12)*np.sqrt(ann)),2),
          'TopEx_ann':round(float(tev.mean()*ann),3),
          'TopEx_Sharpe':round(float(tev.mean()/(tev.std()+1e-12)*np.sqrt(ann)),2),
          'n':int(len(r)),'win':round(float((r>0).mean()),3),'avg_bp':round(float(r.mean()*1e4),1),
          'pf':round(float(r[r>0].sum()/max(-r[r<0].sum(),1e-9)),2),'cov':int(np.mean(cov))}
    print(name,json.dumps(res),flush=True)
    return res

DOM=np.zeros((len(sig_days),N),bool)
for si,t in enumerate(sig_days): DOM[si]=own_lim[t]>=2
out={}
out['combo_raw_full']=evaluate(COMBO,'combo_raw_full')
out['combo_neu_full']=evaluate(COMBO_N,'combo_neu_full')
out['combo_raw_dom']=evaluate(COMBO,'combo_raw_dom',dom=DOM)
out['combo_neu_dom']=evaluate(COMBO_N,'combo_neu_dom',dom=DOM)

# B) 活跃度5档 IC 单调性 (raw与中性化后)
def ic_by_activity(S,name):
    tab={seg:{q:[] for q in range(5)} for seg in ['dev','val']}
    for si,t in enumerate(sig_days):
        segn='dev' if seg_mask(t,*DEV) else ('val' if seg_mask(t,*VAL) else None)
        if segn is None: continue
        s=S[si].copy(); s[~tradable[t]]=np.nan; f=fwd[t]
        act=own_lim[t]
        m=~np.isnan(s)&~np.isnan(f)
        if m.sum()<500: continue
        av=act[m]
        edges=np.quantile(av,[0.2,0.4,0.6,0.8])
        qs=np.digitize(av,edges)
        for q in range(5):
            mm=qs==q
            if mm.sum()<80: continue
            x=s[m][mm]; y=f[m][mm]
            xr=np.argsort(np.argsort(x)).astype(float); yr=np.argsort(np.argsort(y)).astype(float)
            xr=(xr-xr.mean())/(xr.std()+1e-12); yr=(yr-yr.mean())/(yr.std()+1e-12)
            tab[segn][q].append(float((xr*yr).mean()))
    r={}
    for segn in ['dev','val']:
        r[segn]={f'q{q}':{'IC':round(float(np.mean(v)),4),'ICIR':round(float(np.mean(v)/(np.std(v)+1e-12)),2)}
                 for q,v in tab[segn].items() if len(v)>10}
    print(name,json.dumps(r),flush=True)
    return r
out['ic_by_activity_raw']=ic_by_activity(COMBO,'ic_by_activity_raw')
out['ic_by_activity_neu']=ic_by_activity(COMBO_N,'ic_by_activity_neu')

# C) 域加权组合三变体 (raw combo 信号)
WSOFT=[]
for si,t in enumerate(sig_days):
    WSOFT.append(np.where(own_lim[t]>=2,1.0,0.3).astype(np.float32))
out['port_baseline_EW']=out['combo_raw_full']
out['port_domain_only']=out['combo_raw_dom']
out['port_softweight']=evaluate(COMBO,'port_softweight',weights=WSOFT)
json.dump(out,open(f'{OUT}/metrics_v9_neutral_domain.json','w'),ensure_ascii=False,indent=1)
print('done')
