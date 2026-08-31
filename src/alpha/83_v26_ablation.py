# -*- coding: utf-8 -*-
"""v51: v26全组件消融. 逐个去掉每个组件, 量化边际贡献.
组件清单:
 锚块5项: 价格gap / zspread / 双确认gap / TE gap / ROE gap
 行为块2项: 扫单卖 / 卖单均规模
 结构块4项: 未触板 / 非同涨掉队 / 价格>=2元 / dROE未恶化
 域: 活跃域(涨停>=2)
 组合: 打分加权/缓冲带/N200
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
src=open(PROJ+'/src/alpha/82_combo_v26.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
def run13(drop=None,wb=0.4,wh=0.4,ws=0.2,nhold=200,ew=False,nobuf=False,nodom=False):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<8: continue
        v=np.maximum(vol20[t]*np.sqrt(20),1e-4)
        parts=[]
        if drop!='a_price': parts.append(rank_xs(PG_b[t]/v))
        if drop!='a_zs':    parts.append(rank_xs(ZS_b[t]))
        if drop!='a_dual':  parts.append(rank_xs(PG_dual[t]/v))
        if drop!='a_te':    parts.append(rank_xs(PG_te[t]/v))
        if drop!='a_roe':   parts.append(rank_xs(fgap(ROE,t)))
        stk=np.stack(parts)
        with np.errstate(all='ignore'):
            base=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t]) if not nodom else tradable[t]&~np.isnan(PG_b[t])
        base=np.where(dom,base,np.nan)
        hard=(paused[t]<0.5)&(at_hl[t]<0.5)&dom&~np.isnan(base)
        if hard.sum()<100: holdings={}; continue
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        fm=dict(zip(FN,tfeat(t)))
        bl=[]
        if drop!='b_sweep': bl.append(rank01(-fm['sweep_sell'],hard))
        if drop!='b_osize': bl.append(rank01(-fm['osize_sell'],hard))
        behav_r=np.nanmean(np.stack(bl),0) if bl else np.full(N,0.5,np.float32)
        sl=[]
        if drop!='s_board': sl.append(((hl5[t]<1)&(ll5[t]<1)).astype(np.float32))
        if drop!='s_quad':  sl.append((~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))).astype(np.float32))
        if drop!='s_px':    sl.append((np.nan_to_num(close_raw[t],nan=0)>=2.0).astype(np.float32))
        if drop!='s_droe':
            dr=DROE[t]; fin=hard&np.isfinite(dr)
            good=(np.nan_to_num(dr,nan=0)>=np.nanquantile(dr[fin],1/3)).astype(np.float32) if fin.sum()>200 else np.ones(N,np.float32)
            sl.append(good)
        struct_r=np.mean(np.stack(sl),0) if sl else np.full(N,0.5,np.float32)
        with np.errstate(all='ignore'):
            comb=wb*rank01(base,hard)+wh*np.nan_to_num(behav_r,nan=0.5)+ws*struct_r
        comb=np.where(hard,comb,np.nan)
        order=np.argsort(-np.nan_to_num(comb,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        bufmul=1.0 if nobuf else 3.0
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=bufmul*nhold) or np.isnan(comb[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=nhold: break
            if i2 in new_h or np.isnan(comb[i2]) or not hard[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        w=np.ones(len(new_h)) if ew else np.array([np.nan_to_num(comb[i2],nan=0.5)+0.3 for i2 in new_h])
        w=w/w.sum(); wt=dict(zip(new_h,w))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=sum(ww*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,ww in wt.items())-turn*COST
        bq=np.exp(lgq[t_next]-lgq[t])-1
        recs.append((dnum[t],pr,bq,turn/2)); holdings=wt
    return recs
res={}
res['v26_完整']=ev3(run13(),'v26_完整')
for d,nm in [('a_price','去-价格gap锚'),('a_zs','去-zspread'),('a_dual','去-双确认锚'),
             ('a_te','去-TE锚'),('a_roe','去-ROEgap锚'),
             ('b_sweep','去-扫单卖'),('b_osize','去-卖单均规模'),
             ('s_board','去-触板'),('s_quad','去-象限'),('s_px','去-低价'),('s_droe','去-dROE恶化')]:
    res[nm]=ev3(run13(drop=d),nm)
res['去-活跃域']=ev3(run13(nodom=True),'去-活跃域')
res['去-打分加权(等权)']=ev3(run13(ew=True),'去-打分加权(等权)')
res['去-缓冲带']=ev3(run13(nobuf=True),'去-缓冲带')
json.dump(res,open(f'{OUT}/metrics_v51_ablation.json','w'),ensure_ascii=False,indent=1)
print('done')
