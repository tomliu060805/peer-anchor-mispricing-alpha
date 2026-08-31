# -*- coding: utf-8 -*-
"""v20c: v20 相对普通基线的增量.
产品层(同配置换信号): v20信号 vs -mom20 vs -mom20/vol20 vs 仅zspread
信号层(新口径): v20信号 域内IC / Barra(GYCNE5十风格+行业)中性化残差IC / 对rev正交残差IC."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, BARRA_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("res={}")[0]
exec(head)
STYLES=['BETA','BTOP','EARNYILD','GROWTH','LEVERAGE','LIQUIDTY','MOMENTUM','RESVOL','SIZE','SIZENL']
code_ix={c:i for i,c in enumerate(codes)}
# ---- 产品层: 换信号 ----
REV=np.where(~np.isnan(mom20),-mom20,np.nan).astype(np.float32)
REVV=np.where(~np.isnan(mom20),-mom20/np.maximum(vol20*np.sqrt(20),1e-4),np.nan).astype(np.float32)
NANS=np.full((T,N),np.nan,np.float32)
S['rev']=(REV,PM_b,DSP_b,NANS)       # gap槽放rev, zspread槽空
S['revv']=(REVV,PM_b,DSP_b,NANS)
S['zs_only']=(np.where(~np.isnan(S['price'][3]),1.0,np.nan).astype(np.float32),PM_b,DSP_b,S['price'][3])
def ev2(rh,name):
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
res['P_v20']=ev2(run(wmode='score',volnorm=True),'P_v20')
res['P_rev']=ev2(run(sigset='rev',wmode='score'),'P_rev')
res['P_rev_vol']=ev2(run(sigset='revv',wmode='score'),'P_rev_vol')
res['P_zs_only']=ev2(run(sigset='zs_only',wmode='score'),'P_zs_only')
# gap单信号(不含zspread)
S['gap_only']=(S['price'][0],PM_b,DSP_b,NANS)
res['P_gap_only']=ev2(run(sigset='gap_only',wmode='score',volnorm=True),'P_gap_only')

# ---- 信号层 ----
sigd=[t for t in sig5 if dnum[t]>=np.datetime64('2016-01-01')]
def load_expo(i):
    t=sigd[i]; dprev=str(dates[t-1])
    out=np.full((N,len(STYLES)),np.nan,np.float32)
    for j,sty in enumerate(STYLES):
        f=f'{BARRA_ROOT}/{sty}/{dprev}.parquet'
        if not os.path.exists(f): return i,out
        d=pd.read_parquet(f)
        ix=np.array([code_ix.get(c,-1) for c in d['code']])
        v=d[sty].values.astype(np.float32); okm=ix>=0
        out[ix[okm],j]=v[okm]
    return i,out
with ProcessPoolExecutor(50) as ex:
    eo=list(ex.map(load_expo,range(len(sigd)),chunksize=4))
EXPO=np.full((len(sigd),N,len(STYLES)),np.nan,np.float32)
for i,e in eo: EXPO[i]=e
print('expo loaded',flush=True)
def industry_arr(t):
    dstr=str(dates[t-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    return np.array([imap.get(c,'') for c in codes])
def seg_of(t):
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))
acc={s:{k:[] for k in ['ic','ic_neu','ic_orth_rev','ic_rev']} for s in ['dev','val']}
PGv,_,_,ZSv=S['price']
for i,t in enumerate(sigd):
    s0=seg_of(t)
    if s0 is None: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PGv[t])
    gap=PGv[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
    a,b2=rank_xs(np.where(dom,gap,np.nan)),rank_xs(np.where(dom,ZSv[t],np.nan))
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        comb=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
    f=np.exp(logc[min(t+5,T-1)]-logc[t])-1
    cr=rank_xs(comb); rr=rank_xs(np.where(dom,-mom20[t],np.nan)); yr=rank_xs(np.where(dom,f,np.nan))
    m2=~np.isnan(cr)&~np.isnan(yr)&~np.isnan(rr)
    if m2.sum()<300: continue
    c,r0,y=cr[m2],rr[m2],yr[m2]
    c=(c-c.mean())/(c.std()+1e-12); r0=(r0-r0.mean())/(r0.std()+1e-12); y=(y-y.mean())/(y.std()+1e-12)
    acc[s0]['ic'].append(float((c*y).mean()))
    acc[s0]['ic_rev'].append(float((r0*y).mean()))
    co=c-(c*r0).mean()*r0; co/=(co.std()+1e-12)
    acc[s0]['ic_orth_rev'].append(float((co*y).mean()))
    E=EXPO[i]; D=industry_arr(t)
    m3=m2&~np.isnan(E).any(1)
    if m3.sum()>300:
        uniq=[u for u in np.unique(D[m3]) if u]
        Dd=np.zeros((m3.sum(),len(uniq)),np.float32)
        dd=D[m3]
        for j,u in enumerate(uniq): Dd[dd==u,j]=1
        X=np.column_stack([np.ones(m3.sum()),E[m3],Dd])
        keep=X.std(0)>1e-9; keep[0]=True
        cc=rank_xs(comb)[m3]; yy=rank_xs(np.where(dom,f,np.nan))[m3]
        b3=np.linalg.lstsq(X[:,keep],cc,rcond=None)[0]
        resid=cc-X[:,keep]@b3
        resid=(resid-resid.mean())/(resid.std()+1e-12)
        yy=(yy-yy.mean())/(yy.std()+1e-12)
        acc[s0]['ic_neu'].append(float((resid*yy).mean()))
for s0 in ['dev','val']:
    res[f'IC_{s0}']={k:{'IC':round(float(np.mean(v)),4),'ICIR':round(float(np.mean(v)/(np.std(v)+1e-12)),3),'t':round(tstat(v),2)}
                     for k,v in acc[s0].items()}
    print(f'IC_{s0}',json.dumps(res[f'IC_{s0}']),flush=True)
json.dump(res,open(f'{OUT}/metrics_v20c_vs_plain.json','w'),ensure_ascii=False,indent=1)
print('done')
