# -*- coding: utf-8 -*-
"""v47(实验A): 转移熵有向网络. 相关筛top30候选 -> 对每条边算TE(j->i) -> 取TE最高5条为有向邻居.
离散化: 自身状态按域内截面3分位{1,0,-1}; 邻居状态按相对域均{1,-1}; 窗口120周? 用日频120日周频序列.
三种用法: 替换锚/第四锚/收缩因子"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import json, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
st_g=np.load(f'{CACHE}/st_grid.npz')['is_st']
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
W,K,KC=120,5,30
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=[int(x) for x in z8['rebuilds']]
def te_one(args):
    t1,=args if isinstance(args,tuple) else (args,)
    Rw=ret[t1-W:t1]
    valid=(~np.isnan(Rw)).sum(0)>=110
    valid&=~(st_g[t1-1]==1)
    dstr=str(dates[t1-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    inds=np.array([imap.get(c,'') for c in codes])
    valid&=inds!=''
    idx=np.where(valid)[0]; n=len(idx)
    if n<200: return t1,None,None
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
    part=np.argpartition(-C,KC,axis=1)[:,:KC]   # (n,30) 候选
    # 状态离散: 每日截面3分位
    q1=np.quantile(Xr,1/3,axis=1,keepdims=True); q2=np.quantile(Xr,2/3,axis=1,keepdims=True)
    S=np.where(Xr>q2,1,np.where(Xr<q1,-1,0)).astype(np.int8)   # (W,n)
    # TE(j->i): I(Y_t ; X_{t-1} | Y_{t-1}), 3x3x3 计数
    Y=S[1:]; Yl=S[:-1]; L=len(Y)
    te=np.zeros((n,KC),np.float32)
    for c in range(KC):
        j=part[:,c]
        Xl=S[:-1][:,j]                    # (L,n) 邻居滞后状态
        yi=Y; yli=Yl
        # 编码 y*9 + yl*3 + xl  (映射-1,0,1 -> 0,1,2)
        a=(yi+1).astype(np.int32); b=(yli+1).astype(np.int32); cc=(Xl+1).astype(np.int32)
        code3=a*9+b*3+cc
        cnt=np.zeros((n,27),np.float64)
        for k in range(27):
            cnt[:,k]=(code3==k).sum(0)
        p=cnt/L
        p3=p.reshape(n,3,3,3)             # [y,yl,xl]
        p_ylxl=p3.sum(1)                  # (n,3,3)
        p_yyl=p3.sum(3)                   # (n,3,3)
        p_yl=p3.sum((1,3))                # (n,3)
        with np.errstate(all='ignore'):
            num=p3*p_yl[:,None,:,None]
            den=p_yyl[:,:,:,None]*p_ylxl[:,None,:,:]
            r=np.where((p3>0)&(den>0),p3*np.log(np.maximum(num,1e-300)/np.maximum(den,1e-300)),0.0)
        te[:,c]=r.sum((1,2,3))
    ordr=np.argsort(-te,axis=1)[:,:K]
    rows=np.arange(n)[:,None]
    nb=part[rows,ordr]; wv=te[rows,ordr]
    on=np.full((N,K),-1,np.int32); ow=np.zeros((N,K),np.float32)
    on[idx]=idx[nb]; ow[idx]=np.maximum(wv,0)
    return t1,on,ow
if not os.path.exists(f'{CACHE}/nets_v47_te.npz'):
    with ProcessPoolExecutor(45) as ex:
        outs=list(ex.map(te_one,RB8,chunksize=1))
    outs=[o for o in outs if o[1] is not None]
    np.savez_compressed(f'{CACHE}/nets_v47_te.npz',
        t=np.array([o[0] for o in outs],np.int32),
        n=np.stack([o[1] for o in outs]), w=np.stack([o[2] for o in outs]))
    print('TE nets built',len(outs),flush=True)
print('TE net ready',flush=True)
