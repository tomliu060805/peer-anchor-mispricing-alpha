# -*- coding: utf-8 -*-
"""v42(方向3): 日内锚. 网络仍用日频建(不碰v21判负的日内建网), 只把gap的时间尺度换成日内.
 先做纯统计检验: 日内gap(当日/近2日/近3日 兄弟-自身 日内收益差) 对 h=1/2/3/5日 前向的IC与多头超额
 再做产品变体: 日内gap 与 20日gap 混合入锚块"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/68_behav_decay.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']; NB8,WV8=z8['nb_p'],z8['wv_p']
C30=np.load(f'{CACHE}/c30_grid.npz')['c']   # (T,8,N) bar收盘价
close_raw=g['close']
with np.errstate(all='ignore'):
    intr=np.log(C30[:,7,:]/C30[:,0,:])      # 当日开盘bar->收盘bar 的日内收益(不含隔夜)
intr=np.where(np.isfinite(intr)&(np.abs(intr)<0.3),intr,np.nan).astype(np.float32)
ci=np.nancumsum(np.nan_to_num(intr,nan=0.0),0)
def intr_sum(t,d): return ci[t]-ci[t-d] if t-d>=0 else np.full(N,np.nan,np.float32)
def gap_intr(t,d):
    b=np.searchsorted(RB8,t,side='right')-1
    nbr=NB8[b]; wgt=np.maximum(WV8[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    m=intr_sum(t,d)
    nb=nbr[idx]; w0=wgt[idx]
    pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=np.maximum(w.sum(1),1e-9)
    mu=(np.nan_to_num(pm)*w).sum(1)/sw
    out=np.full(N,np.nan,np.float32)
    ok=w.sum(1)>1e-9
    out[idx[ok]]=mu[ok]-m[idx[ok]]
    return out
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
res={}
# A) 统计检验
for d in [1,2,3]:
    acc={s:{h:[] for h in [1,2,3,5]} for s in ['dev','val']}
    lg={s:{h:[] for h in [1,2,3,5]} for s in ['dev','val']}
    for t in sig5:
        s0=seg_of(t)
        if s0 is None or t+6>=T: continue
        gv=gap_intr(t,d)
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(gv)
        ix=np.where(dom)[0]
        if len(ix)<400: continue
        x=gv[ix]
        for h in [1,2,3,5]:
            f=np.exp(logc[t+h,ix]-logc[t,ix])-1
            xr=np.argsort(np.argsort(x)).astype(float); yr=np.argsort(np.argsort(f)).astype(float)
            xr=(xr-xr.mean())/(xr.std()+1e-12); yr=(yr-yr.mean())/(yr.std()+1e-12)
            acc[s0][h].append(float((xr*yr).mean()))
            thr=np.quantile(x,0.9)
            lg[s0][h].append(float(np.nanmean(f[x>=thr])-np.nanmean(f)))
    for s0 in ['dev','val']:
        res[f'A_d{d}_{s0}']={f'h{h}':{'IC':round(float(np.mean(v)),4),'IC_t':round(tstat(v),2),
                                      'long_bp':round(float(np.mean(lg[s0][h]))*1e4,1),'long_t':round(tstat(lg[s0][h]),2)}
                             for h,v in acc[s0].items() if len(v)>20}
        print(f'A_d{d}_{s0}',json.dumps(res[f'A_d{d}_{s0}']),flush=True)
json.dump(res,open(f'{OUT}/metrics_v42_intraday.json','w'),ensure_ascii=False,indent=1)
print('stat done')
