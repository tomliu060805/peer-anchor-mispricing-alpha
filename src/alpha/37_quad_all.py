# -*- coding: utf-8 -*-
"""v20e: 新口径(2016起)四象限全检: 信号层象限表 + 产品层逐象限剔除."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
NHOLD,BM=200,3.0
PG_b,PM_b,DSP_b,ZS_b=S['price']
QN=['P+S+','P+S-','P-S-','P-S+']
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
def ic_of(x,y):
    xr=np.argsort(np.argsort(x)).astype(float); yr=np.argsort(np.argsort(y)).astype(float)
    xr=(xr-xr.mean())/(xr.std()+1e-12); yr=(yr-yr.mean())/(yr.std()+1e-12)
    return float((xr*yr).mean())
res={}
ic_acc={s:{q:[] for q in QN} for s in ['dev','val']}
lg_acc={s:{q:{'ex':[],'cnt':0} for q in QN} for s in ['dev','val']}
tot={s:0 for s in ['dev','val']}
for t in sig5:
    s0=seg_of(t)
    if s0 is None: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    dm=np.nanmean(mom20[t,ix])
    pm=PM_b[t,ix]-dm; om=mom20[t,ix]-dm
    gv=PG_b[t,ix]/np.maximum(vol20[t,ix]*np.sqrt(20),1e-4)
    f=np.exp(logc[min(t+5,T-1),ix]-logc[t,ix])-1
    fu=np.nanmean(f)
    quad=np.where(pm>=0,np.where(om>=0,0,1),np.where(om<0,2,3))
    for qi,q in enumerate(QN):
        m2=quad==qi
        if m2.sum()>=60: ic_acc[s0][q].append(ic_of(gv[m2],f[m2]))
    thr=np.quantile(gv,0.9); top=gv>=thr
    tot[s0]+=int(top.sum())
    for qi,q in enumerate(QN):
        m2=top&(quad==qi)
        lg_acc[s0][q]['cnt']+=int(m2.sum())
        if m2.sum()>=3: lg_acc[s0][q]['ex'].append(float(np.nanmean(f[m2])-fu))
for s0 in ['dev','val']:
    res[f'sig_{s0}']={q:{'IC':round(float(np.mean(v)),4) if len(v)>20 else None,
        'IC_t':round(tstat(v),2) if len(v)>20 else None,
        'long_ex_bp':round(float(np.mean(lg_acc[s0][q]['ex']))*1e4,1) if len(lg_acc[s0][q]['ex'])>20 else None,
        'long_t':round(tstat(lg_acc[s0][q]['ex']),2) if len(lg_acc[s0][q]['ex'])>20 else None,
        'share%':round(lg_acc[s0][q]['cnt']/max(tot[s0],1)*100,1)}
        for q,v in ic_acc[s0].items()}
    print(f'sig_{s0}',json.dumps(res[f'sig_{s0}'],ensure_ascii=False),flush=True)
def run_ex(exclude=None):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5): break
        gap=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZS_b[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        if exclude is not None:
            pmr=PM_b[t]-dm; omr=mom20[t]-dm
            qmask={'P+S+':(pmr>=0)&(omr>=0),'P+S-':(pmr>=0)&(omr<0),
                   'P-S-':(pmr<0)&(omr<0),'P-S+':(pmr<0)&(omr>=0)}[exclude]
            selq&=~qmask
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=BM*NHOLD) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        sc=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc/=sc.sum()
        wt=dict(zip(new_h,sc))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq)); holdings=wt
    return recs
def ev(recs,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
    ex=(1+pr)/(1+bq)-1; ann=252/5
    r={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',FULL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]; nav=np.cumprod(1+e)
        r[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2)}
    print(name,json.dumps(r),flush=True)
    return r
res['prod_no_exclude']=ev(run_ex(None),'prod_no_exclude')
for q in QN:
    res[f'prod_ex_{q}']=ev(run_ex(q),f'prod_ex_{q}')
json.dump(res,open(f'{OUT}/metrics_v20e_quad_all.json','w'),ensure_ascii=False,indent=1)
print('done')
