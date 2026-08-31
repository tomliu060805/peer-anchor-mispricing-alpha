# -*- coding: utf-8 -*-
"""v12: alpha解剖(活跃域, gap信号).
1 机制: 多头腿超额 = 自己涨回 vs 邻居跌落? 两种进入方式(自超跌A/邻超涨B)对称吗?
2 事件: 多头腿按近5日事件分类(自己跌停/自己涨停回落/邻居涨停带动/无事件)的fwd超额
3 时间: 日频信号 alpha衰减曲线 h=1..20, 延迟入场d=0..3, 周几效应
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,WLIM=120,5,20,250
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); ll=np.nan_to_num(at_ll,nan=0.0)
chl=np.cumsum(hl,0); cll=np.cumsum(ll,0)
own_lim=np.zeros((T,N),np.float32); own_lim[WLIM:]=chl[WLIM:]-chl[:-WLIM]
hl5=np.zeros((T,N),np.float32); hl5[5:]=chl[5:]-chl[:-5]     # 近5日涨停次数(t-4..t)
ll5=np.zeros((T,N),np.float32); ll5[5:]=cll[5:]-cll[:-5]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
t_start=int(rb[0])+1
t_end=int(np.searchsorted(dnum,END,side='right'))

# ---- 日频 gap 信号与配套 ----
PG=np.full((T,N),np.nan,np.float32)       # gap
PM=np.full((T,N),np.nan,np.float32)       # peer mom (加权)
NBH5=np.full((T,N),np.nan,np.float32)     # 邻居近5日涨停总次数
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
    PM[t,idx[ok]]=agg[ok]
    NBH5[t,idx]=hl5[t][np.where(nb>=0,nb,0)].sum(1)
print('daily signal ready',flush=True)

def dom_mask(t):
    return (own_lim[t]>=2)&tradable[t]&~np.isnan(PG[t])
def fwd_h(t,h,ix): return np.exp(logc[min(t+h,T-1),ix]-logc[t,ix])-1
def seg_of(t):
    if dnum[t]>=np.datetime64(DEV[0]) and dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]>=np.datetime64(VAL[0]) and dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a); return float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a)))

sig_days=np.arange(t_start,t_end-10,5)
res={}

# ========== 1 机制分解 (周频多头腿) ==========
mech={s:{'own_ex':[],'peer_ex':[],'modeA_ex':[],'modeB_ex':[],'nA':0,'nB':0} for s in ['dev','val']}
for t in sig_days:
    s=seg_of(t)
    if s is None: continue
    dm=dom_mask(t); ix=np.where(dm)[0]
    if len(ix)<300: continue
    gv=PG[t,ix]
    thr=np.quantile(gv,0.9)
    top=ix[gv>=thr]
    uni_mean=np.nanmean(fwd_h(t,5,ix))
    r_own=fwd_h(t,5,top)-uni_mean
    # 邻居前向收益(加权)
    b=np.searchsorted(rb,t,side='right')-1
    nb=NB_P[b][top]; wv=np.maximum(WV_P[b][top],0)
    pfwd=np.exp(logc[min(t+5,T-1)][np.where(nb>=0,nb,0)]-logc[t][np.where(nb>=0,nb,0)])-1
    msk=(nb>=0); wv=wv*msk
    r_peer=(np.nan_to_num(pfwd)*wv).sum(1)/np.maximum(wv.sum(1),1e-9)-uni_mean
    mech[s]['own_ex'].append(float(np.nanmean(r_own)))
    mech[s]['peer_ex'].append(float(np.nanmean(r_peer)))
    # 进入方式: 域均动量
    dmean=np.nanmean(mom[t,ix])
    own_rel=mom[t,top]-dmean; peer_rel=PM[t,top]-dmean
    A=(-own_rel)>peer_rel     # 自己超跌主导
    mech[s]['modeA_ex'].append(float(np.nanmean(r_own[A])) if A.any() else np.nan)
    mech[s]['modeB_ex'].append(float(np.nanmean(r_own[~A])) if (~A).any() else np.nan)
    mech[s]['nA']+=int(A.sum()); mech[s]['nB']+=int((~A).sum())
for s in ['dev','val']:
    d=mech[s]
    res[f'mech_{s}']={
      'own_ex_bp':round(np.nanmean(d['own_ex'])*1e4,1),'own_t':round(tstat([x for x in d['own_ex'] if x==x]),2),
      'peer_ex_bp':round(np.nanmean(d['peer_ex'])*1e4,1),'peer_t':round(tstat([x for x in d['peer_ex'] if x==x]),2),
      'modeA_own超跌_bp':round(np.nanmean(d['modeA_ex'])*1e4,1),'modeA_t':round(tstat([x for x in d['modeA_ex'] if x==x]),2),
      'modeB_邻超涨_bp':round(np.nanmean(d['modeB_ex'])*1e4,1),'modeB_t':round(tstat([x for x in d['modeB_ex'] if x==x]),2),
      'nA':d['nA'],'nB':d['nB']}
print('mech done',json.dumps(res,ensure_ascii=False),flush=True)

# ========== 2 事件条件化 (周频多头腿) ==========
ev={s:{k:[] for k in ['E1自跌停','E2自涨停回落','E3邻涨停带动','E0无事件']} for s in ['dev','val']}
for t in sig_days:
    s=seg_of(t)
    if s is None: continue
    dm=dom_mask(t); ix=np.where(dm)[0]
    if len(ix)<300: continue
    gv=PG[t,ix]; thr=np.quantile(gv,0.9)
    top=ix[gv>=thr]
    uni_mean=np.nanmean(fwd_h(t,5,ix))
    r=fwd_h(t,5,top)-uni_mean
    e1=ll5[t,top]>=1
    e2=(hl5[t,top]>=1)&~e1
    e3=(NBH5[t,top]>=2)&~e1&~e2
    e0=~e1&~e2&~e3
    for k,m2 in [('E1自跌停',e1),('E2自涨停回落',e2),('E3邻涨停带动',e3),('E0无事件',e0)]:
        if m2.sum()>=3: ev[s][k].append(float(np.nanmean(r[m2])))
for s in ['dev','val']:
    res[f'event_{s}']={k:{'bp':round(np.nanmean(v)*1e4,1),'t':round(tstat(v),2),'n':len(v)}
                       for k,v in ev[s].items() if len(v)>10}
print('event done',flush=True)

# ========== 3 时间结构 (日频信号) ==========
# 3a 衰减: 多头腿(日频 top10%) 累计超额 vs h; 3b 延迟入场 d; 3c 周几
decay={s:{h:[] for h in [1,2,3,5,10,15,20]} for s in ['dev','val']}
delay={s:{d2:[] for d2 in [0,1,2,3]} for s in ['dev','val']}
wday={s:{} for s in ['dev','val']}
for t in range(t_start,t_end-25):
    s=seg_of(t)
    if s is None: continue
    dm=dom_mask(t); ix=np.where(dm)[0]
    if len(ix)<300: continue
    gv=PG[t,ix]; thr=np.quantile(gv,0.9)
    top=ix[gv>=thr]
    for h in decay[s]:
        u=np.nanmean(fwd_h(t,h,ix))
        decay[s][h].append(float(np.nanmean(fwd_h(t,h,top))-u))
    for d2 in delay[s]:
        t2=t+d2
        u=np.nanmean(np.exp(logc[t2+5,ix]-logc[t2,ix])-1)
        v=np.nanmean(np.exp(logc[t2+5,top]-logc[t2,top])-1)
        delay[s][d2].append(float(v-u))
    wd=int(dnum[t].astype('datetime64[D]').astype(object).weekday())
    u=np.nanmean(fwd_h(t,5,ix))
    wday[s].setdefault(wd,[]).append(float(np.nanmean(fwd_h(t,5,top))-u))
for s in ['dev','val']:
    res[f'decay_{s}']={f'h{h}':{'bp':round(np.mean(v)*1e4,1),'t':round(tstat(v),2)} for h,v in decay[s].items()}
    res[f'delay_{s}']={f'd{d2}':{'bp':round(np.mean(v)*1e4,1),'t':round(tstat(v),2)} for d2,v in delay[s].items()}
    res[f'weekday_{s}']={['周一','周二','周三','周四','周五'][w]:{'bp':round(np.mean(v)*1e4,1),'n':len(v)}
                         for w,v in sorted(wday[s].items())}
json.dump(res,open(f'{OUT}/metrics_v12_anatomy.json','w'),ensure_ascii=False,indent=1)
for k,v in res.items(): print(k,json.dumps(v,ensure_ascii=False))
print('done')
