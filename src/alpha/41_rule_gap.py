# -*- coding: utf-8 -*-
"""v22b: 主规则(信号日收盘执行) vs 后备规则(T+1 11:30卖/13:30买) 完整净值对比.
后备口径: 卖出腿 close_t->P(t+1,11:30)后转现金; 买入腿 P(t+1,13:30)入场->持有;
续持腿不变. 复权: 日内腿用原始价(隔夜|跳|>15%回退收盘执行), 多日腿用复权链."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
close_raw=g['close']
C30=np.load(f'{CACHE}/c30_grid.npz')['c']
NHOLD,BM=200,3.0
PG_b,PM_b,DSP_b,ZS_b=S['price']
BAR_SELL,BAR_BUY=3,4   # 11:30, 13:30
def replay(mode):
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
        selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings); sold=[]
        for i2 in list(new_h):
            if ((rank[i2]>=BM*NHOLD) or np.isnan(s[i2])) and can_sell[i2]:
                sold.append((i2,new_h[i2])); del new_h[i2]
        bought=[]
        for i2 in order:
            if len(new_h)>=NHOLD: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0; bought.append(i2)
        if len(new_h)<40: holdings={}; continue
        sc=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc/=sc.sum()
        wt=dict(zip(new_h,sc))
        turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
        t_next=sig5[si+1]
        pr=0.0
        if mode=='main':
            pr=sum(w*(np.exp(logc[t_next,i2]-logc[t,i2])-1) for i2,w in wt.items())
        else:
            bset=set(bought)
            for i2,w in wt.items():
                if i2 in bset and t+1<T:
                    p_in=C30[t+1,BAR_BUY,i2]; c1=close_raw[t+1,i2]; c0=close_raw[t,i2]
                    if p_in==p_in and c1==c1 and c0==c0 and c0>0 and abs(C30[t+1,0,i2]/c0-1)<=0.15:
                        r=(c1/p_in)*np.exp(logc[t_next,i2]-logc[t+1,i2])-1
                    else:
                        r=np.exp(logc[t_next,i2]-logc[t,i2])-1
                else:
                    r=np.exp(logc[t_next,i2]-logc[t,i2])-1
                pr+=w*r
            # 卖出腿: close_t -> t+1 11:30, 之后现金(占先前权重, 对本期收益的贡献差额)
            for i2,w_old in sold:
                c0=close_raw[t,i2]; p_out=C30[t+1,BAR_SELL,i2] if t+1<T else np.nan
                if p_out==p_out and c0==c0 and c0>0 and abs(C30[t+1,0,i2]/c0-1)<=0.15:
                    pr+=w_old*(p_out/c0-1)   # 主规则中该腿为0(t收盘即离场), 此为差异项
        pr-=turn*COST
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
    m=(ds>=np.datetime64(FULL[0]))&(ds<=np.datetime64(FULL[1]))
    years=sorted(set(str(d)[:4] for d in ds[m]))
    r['yearly']={y:round(float(np.prod(1+ex[m][[str(x)[:4]==y for x in ds[m]]])-1),4) for y in years}
    print(name,json.dumps(r),flush=True)
    return r
res={}
res['main']=ev(replay('main'),'main')
res['backup']=ev(replay('backup'),'backup')
json.dump(res,open(f'{OUT}/metrics_v22b_rulegap.json','w'),ensure_ascii=False,indent=1)
print('done')
