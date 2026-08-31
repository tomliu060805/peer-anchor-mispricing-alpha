# -*- coding: utf-8 -*-
"""v22: 日内买入时点结构. 重放v20调仓记录买/卖名单, 用30min bar收盘价测:
A) 相对信号日收盘的执行成本曲线 rel_j = mean log(P_{t+1,bar j}/close_t)  (买入: 越低越好)
B) 各bar入场至t+5收盘的持有收益差
C) 卖出名单同curve (越高越好卖)
分 dev/val; 剔除隔夜|跳|>15%的复权污染样本."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, STOCK_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
close_raw=g['close']
code_ix={c:i for i,c in enumerate(codes)}
# ---- C30 bar收盘价缓存 ----
def load_day30(i):
    f=f'{STOCK_ROOT}/price/price_30m/{dates[i]}.parquet'
    out=np.full((8,N),np.nan,np.float32)
    if not os.path.exists(f): return i,out
    d=pd.read_parquet(f,columns=['datetime','code','close'])
    bars=sorted(d['datetime'].unique())[:8]
    bmap={b:j for j,b in enumerate(bars)}
    ix=np.array([code_ix.get(c,-1) for c in d['code']])
    bj=np.array([bmap.get(b,-1) for b in d['datetime']])
    v=d['close'].values.astype(np.float32)
    okm=(ix>=0)&(bj>=0)
    out[bj[okm],ix[okm]]=v[okm]
    return i,out
if not os.path.exists(f'{CACHE}/c30_grid.npz'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(load_day30,range(T),chunksize=8))
    C30=np.full((T,8,N),np.nan,np.float32)
    for i,o in outs: C30[i]=o
    np.savez_compressed(f'{CACHE}/c30_grid.npz',c=C30)
    print('c30 cached',flush=True)
C30=np.load(f'{CACHE}/c30_grid.npz')['c']
print('C30 loaded',flush=True)
# ---- 重放 v20 记录交易 ----
NHOLD,BM=200,3.0
PG_b,PM_b,DSP_b,ZS_b=S['price']
buys=[]; sells=[]
holdings={}
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
    selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
    order=np.argsort(-np.nan_to_num(s,nan=-1e9))
    rank=np.full(N,1<<30); rank[order]=np.arange(N)
    can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
    new_h=dict(holdings)
    for i2 in list(new_h):
        if ((rank[i2]>=BM*NHOLD) or np.isnan(s[i2])) and can_sell[i2]:
            del new_h[i2]; sells.append((si,t,i2))
    for i2 in order:
        if len(new_h)>=NHOLD: break
        if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
        new_h[i2]=0.0; buys.append((si,t,i2))
    if len(new_h)<40: holdings={}; continue
    holdings={i2:1.0/len(new_h) for i2 in new_h}
print(f'buys={len(buys)} sells={len(sells)}',flush=True)
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def curve(trades,name):
    acc={s0:{j:[] for j in range(8)} for s0 in ['dev','val']}
    hold_acc={s0:{j:[] for j in range(8)} for s0 in ['dev','val']}
    for si,t,i2 in trades:
        s0=seg_of(t)
        if s0 is None or t+6>=T: continue
        c0=close_raw[t,i2]
        p=C30[t+1,:,i2]
        if not (c0==c0 and c0>0) or np.isnan(p).any(): continue
        ov=p[0]/c0-1
        if abs(ov)>0.15: continue    # 复权/异常剔除
        rel=np.log(p/c0)
        for j in range(8): acc[s0][j].append(float(rel[j]))
        # 持有: bar j 买入 -> t+5收盘(复权链: t+1收盘->t+5收盘 用logc)
        tail=np.exp(logc[min(t+6,T-1),i2]-logc[t+1,i2])
        for j in range(8):
            r=(p[7]/p[j])*tail-1.0
            hold_acc[s0][j].append(float(r))
    out={}
    for s0 in ['dev','val']:
        out[s0]={'exec_bp':[round(float(np.mean(acc[s0][j]))*1e4,1) for j in range(8)],
                 'hold_bp':[round(float(np.mean(hold_acc[s0][j]))*1e4,1) for j in range(8)],
                 'n':len(acc[s0][0])}
    print(name,json.dumps(out),flush=True)
    return out
res={}
res['buy']=curve(buys,'buy')
res['sell']=curve(sells,'sell')
json.dump(res,open(f'{OUT}/metrics_v22_entry.json','w'),ensure_ascii=False,indent=1)
print('done')
