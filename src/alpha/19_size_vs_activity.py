# -*- coding: utf-8 -*-
"""v14: 小盘均值回归 vs 活跃度机制 拆解.
活跃度两定义: limA=250日收盘封涨停>=2; altA=250日内日涨幅>=8%的天数>=2 (放宽,大票可达)
1) 覆盖诊断: 300/500/1000/其余 各层通过 limA/altA 的比例
2) 网格: 市值层(300/500/1000/其余) × altA(是/否) 的 gap IC 与 多头20%超额(相对格内均值)
3) 300∩altA / 500∩altA 域内 完整信号(combo) IC/多头超额
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, INDEX_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np, pandas as pd

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM,HOLD=120,5,20,250,5
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); chl=np.cumsum(hl,0)
lim250=np.zeros((T,N),np.float32); lim250[WLIM:]=chl[WLIM:]-chl[:-WLIM]
big8=(np.nan_to_num(ret,nan=0.0)>=0.08).astype(np.float32); cb8=np.cumsum(big8,0)
alt250=np.zeros((T,N),np.float32); alt250[WLIM:]=cb8[WLIM:]-cb8[:-WLIM]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
t_start=int(rb[0])+1
t_end=int(np.searchsorted(dnum,END,side='right'))
code_ix={c:i for i,c in enumerate(codes)}

# 成分掩码 (信号日读取)
def mem_mask(pool,t):
    for lag in range(0,6):
        f=f'{INDEX_ROOT}/weight/{pool}/{dates[t-lag]}.parquet'
        if os.path.exists(f):
            d=pd.read_parquet(f,columns=['stock_code'])
            m=np.zeros(N,bool)
            ix=[code_ix.get(c,-1) for c in d['stock_code']]
            m[[i for i in ix if i>=0]]=True
            return m
    return np.zeros(N,bool)

# 日频 gap + zspread combo (同17)
PG=np.full((T,N),np.nan,np.float32)
for t in range(t_start,t_end):
    b=np.searchsorted(rb,t,side='right')-1
    nbr=NB_P[b]; wgt=np.maximum(WV_P[b],0)
    idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
    nb=nbr[idx]; w0=wgt[idx]
    m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
    msk=(nb>=0)&~np.isnan(pm); w=w0*msk
    sw=w.sum(1); ok=sw>1e-9
    agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
    PG[t,idx[ok]]=agg[ok]-m[idx[ok]]
print('gap ready',flush=True)

def seg_of(t):
    if dnum[t]>=np.datetime64(DEV[0]) and dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]>=np.datetime64(VAL[0]) and dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,dtype=float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
def ic_of(x,y):
    xr=np.argsort(np.argsort(x)).astype(float); yr=np.argsort(np.argsort(y)).astype(float)
    xr=(xr-xr.mean())/(xr.std()+1e-12); yr=(yr-yr.mean())/(yr.std()+1e-12)
    return float((xr*yr).mean())

sig_days=np.arange(t_start,t_end-6,5)
res={}
# 1) 覆盖诊断 (每年抽4个信号日平均)
cov={}
for t in sig_days[::13]:
    m300=mem_mask('csi_300',t); m500=mem_mask('csi_500',t); m1000=mem_mask('csi_1000',t)
    rest=~(m300|m500|m1000)&~np.isnan(ret[t])
    for nm,mm in [('300',m300),('500',m500),('1000',m1000),('other',rest)]:
        n=mm.sum()
        if n==0: continue
        cov.setdefault(nm,[]).append((float((lim250[t][mm]>=2).mean()),float((alt250[t][mm]>=2).mean())))
res['coverage']={nm:{'limA%':round(np.mean([a for a,b in v])*100,1),'altA%':round(np.mean([b for a,b in v])*100,1)}
                 for nm,v in cov.items()}
print('coverage',json.dumps(res['coverage']),flush=True)

# 2) 网格: 市值层 × altA
grid={s:{} for s in ['dev','val']}
# 3) 300∩altA / 500∩altA 完整域测试
dom_res={s:{} for s in ['dev','val']}
for t in sig_days:
    s=seg_of(t)
    if s is None: continue
    m300=mem_mask('csi_300',t); m500=mem_mask('csi_500',t); m1000=mem_mask('csi_1000',t)
    rest=~(m300|m500|m1000)
    base=tradable[t]&~np.isnan(PG[t])
    f=np.exp(logc[min(t+HOLD,T-1)]-logc[t])-1
    act=alt250[t]>=2
    for nm,mm in [('300',m300),('500',m500),('1000',m1000),('other',rest)]:
        for anm,am in [('act',act),('inact',~act)]:
            cell=mm&am&base&~np.isnan(f)
            if cell.sum()<40: continue
            gv=PG[t,cell]; fv=f[cell]
            key=f'{nm}_{anm}'
            d=grid[s].setdefault(key,{'ic':[],'te':[]})
            d['ic'].append(ic_of(gv,fv))
            gq=gv>=np.quantile(gv,0.8)
            d['te'].append(float(fv[gq].mean()-fv.mean()))
for s in ['dev','val']:
    res[f'grid_{s}']={k:{'IC':round(float(np.mean(v['ic'])),4),'IC_t':round(tstat(v['ic']),2),
                         'top_ex_bp':round(float(np.mean(v['te']))*1e4,1),'te_t':round(tstat(v['te']),2),
                         'n':len(v['ic'])}
                      for k,v in grid[s].items() if len(v['ic'])>20}
json.dump(res,open(f'{OUT}/metrics_v14_size_activity.json','w'),ensure_ascii=False,indent=1)
for k in res:
    print(k,json.dumps(res[k],ensure_ascii=False))
print('done')
