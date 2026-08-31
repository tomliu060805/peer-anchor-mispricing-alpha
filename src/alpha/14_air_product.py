# -*- coding: utf-8 -*-
"""v11: 活跃域绝对收益多头(空气指增)产品参数扫描.
域: own_lim250>=2; 信号: raw combo (对照 neu); 规则: 进TopN, 跌出buffer*N才卖(缓冲带)
现实约束: 停牌/涨停不能买, 停牌/跌停不能卖(顺延); 成本按实际换手双边
输出: 年化/波动/Sharpe/MDD/逐年/换手/单笔指标 + 容量估算; 对照000852仅参考
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np, pandas as pd

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,COST,WLIM=120,5,20,0.00125,250
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
money=g['money']; paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); chl=np.cumsum(hl,0)
own_lim=np.zeros((T,N),np.float32); own_lim[WLIM:]=chl[WLIM:]-chl[:-WLIM]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
ir=np.load(f'{CACHE}/idx_ret.npz')['000852_XSHG']; lgi=np.cumsum(np.log1p(ir))
# 20日ADV
cm=np.cumsum(np.nan_to_num(money,nan=0.0),0)
adv20=np.full((T,N),np.nan,np.float32); adv20[20:]=(cm[20:]-cm[:-20])/20

def build_signal(hold):
    sig_days=np.arange(rb[0]+1,T-1-hold,hold)
    S_pg=np.full((len(sig_days),N),np.nan,np.float32)
    S_zs=np.full((len(sig_days),N),np.nan,np.float32)
    for si,t in enumerate(sig_days):
        b=np.searchsorted(rb,t,side='right')-1
        nbr=NB_P[b]; wgt=np.maximum(WV_P[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w=wgt[idx]
        m=mom[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w*msk
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
        S_pg[si,idx[ok]]=agg[ok]-m[idx[ok]]
    for b in range(len(rb)):
        t1=rb[b]; t0=t1-W
        t2=rb[b+1] if b+1<len(rb) else T
        sds=[(si,t) for si,t in enumerate(sig_days) if t1<=t<t2]
        if not sds: continue
        nbr=NB_P[b]; idx=np.where(nbr[:,0]>=0)[0]; nb=nbr[idx]
        base=logc[t0-1] if t0>0 else np.zeros(N)
        Pw=np.exp(logc[t0:t1]-base)
        nbc=np.where(nb>=0,nb,0)
        sj=Pw[:,nbc.ravel()].reshape(W,len(idx),K)
        s_tr=Pw[:,idx][:,:,None]-sj
        mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
        d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
        wsm=np.exp(-d); wsm/=np.maximum(wsm.sum(1,keepdims=True),1e-12)
        for si,t in sds:
            Pt=np.exp(logc[t]-base)
            zv=(Pt[idx][:,None]-Pt[nbc]-mu)/sd2
            S_zs[si,idx]=-np.nansum(zv*wsm,1)
    def rank_xs(x):
        m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
        if m.sum()<50: return r
        rr2=np.argsort(np.argsort(x[m])).astype(np.float32)
        r[m]=(rr2-rr2.mean())/(rr2.std()+1e-12); return r
    C=np.full_like(S_pg,np.nan)
    for si in range(len(sig_days)):
        a,b2=rank_xs(S_pg[si]),rank_xs(S_zs[si])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            C[si]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
    return sig_days,C

SIGS={h:build_signal(h) for h in [5,10]}
print('signals ready',flush=True)

def run(nhold,hold,buffer_mult,cost=COST,weight='ew'):
    sig_days,C=SIGS[hold]
    holdings={}
    recs=[]
    for si,t in enumerate(sig_days):
        if dnum[t]>END: break
        s=C[si].copy()
        dom=(own_lim[t]>=2)&~np.isnan(ret[t])&(st[t]<0.5)
        s[~dom]=np.nan
        can_buy=(paused[t]<0.5)&(at_hl[t]<0.5)
        can_sell=(paused[t]<0.5)&(at_ll[t]>=0)&(at_ll[t]<0.5)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        nvalid=int((~np.isnan(s)).sum())
        # 卖: 跌出 buffer*N 或出域, 且能卖
        new_h=dict(holdings)
        for i2 in list(new_h):
            if (rank[i2]>=buffer_mult*nhold or np.isnan(s[i2])) and can_sell[i2]:
                del new_h[i2]
        # 买: 按rank补到 nhold
        for i2 in order:
            if len(new_h)>=nhold: break
            if rank[i2]>=min(nvalid,1<<29): break
            if i2 in new_h or not can_buy[i2] or np.isnan(s[i2]): continue
            new_h[i2]=0.0
        # 权重
        if weight=='ew':
            wt={i2:1.0/len(new_h) for i2 in new_h}
        else:
            sc=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h])
            sc/=sc.sum(); wt=dict(zip(new_h,sc))
        turn=0.0
        keys=set(wt)|set(holdings)
        for i2 in keys: turn+=abs(wt.get(i2,0)-holdings.get(i2,0))
        pr=sum(w*(np.exp(logc[min(t+hold,T-1),i2]-logc[t,i2])-1) for i2,w in wt.items())
        pr-=turn*cost
        advs=[adv20[t,i2] for i2 in wt if adv20[t,i2]==adv20[t,i2]]
        recs.append((dnum[t],pr,turn/2,np.median(advs) if advs else np.nan))
        holdings=wt
    return recs

def eval_abs(recs,hold,name):
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs])
    tn=np.array([r[2] for r in recs]); ad=np.array([r[3] for r in recs])
    ann=252/hold
    res={}
    for segn,(a,b) in [('dev',DEV),('val',VAL),('full',('2015-01-01','2024-08-16'))]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        r=pr[m]; nav=np.cumprod(1+r)
        bench=[]
        tmap={d:i for i,d in enumerate(dnum)}
        for d in ds[m]:
            t=tmap[d]; bench.append(np.exp(lgi[min(t+hold,T-1)]-lgi[t])-1)
        ex=(1+r)/(1+np.array(bench))-1
        res[segn]={'AnnRet':round(float(nav[-1]**(ann/len(r))-1),4),
          'Vol':round(float(r.std()*np.sqrt(ann)),4),
          'Sharpe':round(float(r.mean()/(r.std()+1e-12)*np.sqrt(ann)),2),
          'MDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),
          'Ex852_ann':round(float(np.prod(1+ex)**(ann/len(ex))-1),4),
          'n':int(len(r)),'win':round(float((r>0).mean()),3),'avg_bp':round(float(r.mean()*1e4),1),
          'pf':round(float(r[r>0].sum()/max(-r[r<0].sum(),1e-9)),2),
          'turn_1side':round(float(tn[m].mean()),3),'medADV_万':round(float(np.nanmedian(ad[m])/1e4),0)}
    years=sorted(set(str(d)[:4] for d in ds))
    res['yearly']={y:round(float(np.prod(1+pr[[str(d)[:4]==y for d in ds]])-1),4) for y in years}
    print(name,json.dumps(res,ensure_ascii=False),flush=True)
    return res

out={}
for nhold in [50,100,200]:
    for bm in [1.0,2.0,3.0]:
        out[f'N{nhold}_H5_B{int(bm)}']=eval_abs(run(nhold,5,bm),5,f'N{nhold}_H5_B{int(bm)}')
out['N100_H10_B2']=eval_abs(run(100,10,2.0),10,'N100_H10_B2')
out['N100_H5_B2_sw']=eval_abs(run(100,5,2.0,weight='score'),5,'N100_H5_B2_sw')
out['N100_H5_B2_cost25']=eval_abs(run(100,5,2.0,cost=0.0025),5,'N100_H5_B2_cost25')
json.dump(out,open(f'{OUT}/metrics_v11_air.json','w'),ensure_ascii=False,indent=1)
print('done')
