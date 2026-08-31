# -*- coding: utf-8 -*-
"""增量检验: 网络信号是否在纯自身反转(-mom20)之外提供 alpha.
1) 基准 rev=-mom20 的 IC
2) 正交化: peer 信号截面回归剔除 rev 后的残差 IC (resid网络 vs random placebo)
3) Fama-MacBeth 双变量秩回归: fwd ~ rev + peer, 看 peer 系数 t
4) 期限扫描: 正交 peer 信号对 fwd 5/10/21/42/63d 的 IC
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, numpy as np

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,REBUILD,K,MOM,HOLD=120,21,10,20,5
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
listed=~np.isnan(ret)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&listed
dnum=dates.astype('datetime64[D]')

sigs={}
for net in ['resid','random','industry']:
    z=np.load(f'{OUT}/signals_{net}.npz')
    sigs[net]=z
sig_days=sigs['resid']['sig_days']

def fwd_k(k):
    f=np.full((T,N),np.nan,np.float32)
    f[:T-k]=(np.exp(logc[k:]-logc[:-k])-1).astype(np.float32)
    return f
FWD={k:fwd_k(k) for k in [5,10,21,42,63]}

def rank_xs(x):
    m=~np.isnan(x)
    r=np.full_like(x,np.nan)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float64)
    rr=(rr-rr.mean())/(rr.std()+1e-12)
    r[m]=rr
    return r

def seg_mask(t,seg): return (dnum[t]>=np.datetime64(seg[0]))and(dnum[t]<=np.datetime64(seg[1]))

def ic_series(S,k=5,seg=DEV,orth_on=None):
    ics=[]
    f=FWD[k]
    for si,t in enumerate(sig_days):
        if not seg_mask(t,seg): continue
        s=S[si].copy(); s[~tradable[t]]=np.nan
        y=f[t]
        sr=rank_xs(s); yr=rank_xs(y)
        if orth_on is not None:
            br=rank_xs(orth_on[si] if orth_on.shape[0]==len(sig_days) else orth_on[t])
            m=~np.isnan(sr)&~np.isnan(br)
            if m.sum()<50: continue
            beta=np.nansum(sr[m]*br[m])/np.nansum(br[m]**2)
            sr=sr-beta*br  # 残差
        m=~np.isnan(sr)&~np.isnan(yr)
        if m.sum()<50: continue
        x=sr[m]; x=(x-x.mean())/(x.std()+1e-12)
        ics.append(float((x*yr[m]).mean()))
    ics=np.array(ics)
    return {'IC':round(float(ics.mean()),4),'ICIR':round(float(ics.mean()/(ics.std()+1e-12)),3),
            't':round(float(ics.mean()/(ics.std()+1e-12)*np.sqrt(len(ics))),2),'n':len(ics)}

def fmb(S,seg=DEV,k=5):
    """fwd_rank ~ rev_rank + peer_rank 截面回归, 返回 peer 系数 t"""
    f=FWD[k]; coefs=[]
    for si,t in enumerate(sig_days):
        if not seg_mask(t,seg): continue
        s=S[si].copy(); s[~tradable[t]]=np.nan
        rv=-mom[t]
        sr,br,yr=rank_xs(s),rank_xs(rv),rank_xs(f[t])
        m=~np.isnan(sr)&~np.isnan(br)&~np.isnan(yr)
        if m.sum()<100: continue
        X=np.stack([np.ones(m.sum()),br[m],sr[m]],1)
        b=np.linalg.lstsq(X,yr[m],rcond=None)[0]
        coefs.append(b)
    C=np.array(coefs)
    out={}
    for i,nm in enumerate(['const','rev','peer']):
        c=C[:,i]; out[nm]={'coef':round(float(c.mean()),4),'t':round(float(c.mean()/(c.std()+1e-12)*np.sqrt(len(c))),2)}
    return out

res={}
# 1) 基准: rev
REV=np.full((len(sig_days),N),np.nan,np.float32)
for si,t in enumerate(sig_days): REV[si]=-mom[t]
for segn,seg in [('dev',DEV),('val',VAL)]:
    res[f'rev:{segn}']=ic_series(REV,5,seg)
print('rev baseline:',json.dumps(res,ensure_ascii=False),flush=True)

# 2) 正交化 IC + 3) FMB + 4) 期限扫描
for net in ['resid','industry','random']:
    for signame in ['peer_mom','peer_gap','zspread']:
        S=sigs[net][signame]
        for segn,seg in [('dev',DEV),('val',VAL)]:
            res[f'{net}:{signame}:orth_rev:{segn}']=ic_series(S,5,seg,orth_on=REV)
        res[f'{net}:{signame}:fmb_dev']=fmb(S,DEV)['peer']
        res[f'{net}:{signame}:fmb_val']=fmb(S,VAL)['peer']
    # 期限扫描只做 peer_mom 正交
    S=sigs[net]['peer_mom']
    for k in [5,10,21,42,63]:
        res[f'{net}:peer_mom:orth:h{k}:dev']=ic_series(S,k,DEV,orth_on=REV)
    print(f'{net} done',flush=True)

with open(f'{OUT}/metrics_v2_incremental.json','w') as fh: json.dump(res,fh,ensure_ascii=False,indent=1)
for k,v in res.items(): print(k,v)
