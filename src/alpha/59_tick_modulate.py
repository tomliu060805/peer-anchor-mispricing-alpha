# -*- coding: utf-8 -*-
"""v31: 逐笔行为调制. 特征(信号日前5日聚合, 均可实时):
 sweep_sell_ratio = 主动卖扫单额/总成交额   (恐慌砸单强度)
 patient_sell     = (主动卖额-扫单额)/总成交额 (耐心派发强度)
 order_size_sell  = 主动卖额/主动卖单数 (卖方单笔规模)
 aggr_imbalance   = (主动买额-主动卖额)/总成交额
A) 多头腿三分位条件表  B) 产品过滤(每特征两方向)  基线=v21"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/48_combo_final.py').read()
head=src.split("res={}")[0]
exec(head)
close_raw=g['close']
code_ix={c:i for i,c in enumerate(codes)}
TB=PROJ+'/cache/tickbehav'
def load_tb(i):
    ds=str(dates[i]); f=f'{TB}/{ds}.parquet'
    out=np.full((6,N),np.nan,np.float32)
    if not os.path.exists(f): return i,out
    d=pd.read_parquet(f)
    suf=np.where(d['exchange'].values=='SZ','.XSHE','.XSHG')
    full=np.char.add(d['code'].values.astype(str),suf)
    ix=np.array([code_ix.get(c,-1) for c in full]); okm=ix>=0
    for j,col in enumerate(['total_money','asell_amt','asell_sweep','asell_orders','abuy_amt','abuy_sweep']):
        v=d[col].values.astype(np.float32) if col in d.columns else np.zeros(len(d),np.float32)
        out[j,ix[okm]]=np.nan_to_num(v[okm],nan=0.0)
    return i,out
if not os.path.exists(f'{CACHE}/tb_grid.npz'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(load_tb,range(T),chunksize=8))
    TBG=np.full((T,6,N),np.nan,np.float32)
    for i,o in outs: TBG[i]=o
    np.savez_compressed(f'{CACHE}/tb_grid.npz',tb=TBG)
    print('tb grid cached',flush=True)
TBG=np.load(f'{CACHE}/tb_grid.npz')['tb']
tbc=np.nancumsum(np.nan_to_num(TBG,nan=0.0),axis=0)
def tfeat(t,lag=0):
    e=t-lag; s=tbc[e]-tbc[e-5]
    tm,sa,ss,so,ba,bs=s
    tm=np.maximum(tm,1.0)
    has=(tbc[e,0]-tbc[e-5,0])>0
    sweep_sell=ss/tm
    patient_sell=(sa-ss)/tm
    osize=np.where(so>0,sa/np.maximum(so,1),np.nan)
    imb=(ba-sa)/tm
    f=[sweep_sell,patient_sell,osize,imb]
    return [np.where(has,x,np.nan) for x in f]
FN=['sweep_sell','patient_sell','osize_sell','aggr_imb']
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
res={}
acc={s:{f:{q:[] for q in range(3)} for f in FN} for s in ['dev','val']}
for t in sig5:
    s0=seg_of(t)
    if s0 is None or t<6: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    a,b2=rank_xs(PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)),rank_xs(ZS_b[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        sc=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))[ix]
    f5=tfeat(t)
    fwd=np.exp(logc[min(t+5,T-1),ix]-logc[t,ix])-1
    fu=np.nanmean(fwd)
    thr=np.nanquantile(sc,0.9); top=sc>=thr
    for fname,fv in zip(FN,f5):
        v=fv[ix]; m2=top&~np.isnan(v)
        if m2.sum()<20: continue
        qe=np.nanquantile(v[m2],[1/3,2/3]); qq=np.digitize(v[m2],qe)
        for q in range(3):
            mm=qq==q
            if mm.sum()>=5: acc[s0][fname][q].append(float(np.nanmean(fwd[m2][mm])-fu))
for s0 in ['dev','val']:
    res[f'A_{s0}']={f:{f'q{q}':{'bp':round(float(np.mean(v))*1e4,1),'t':round(tstat(v),2)} for q,v in d.items() if len(v)>20}
                    for f,d in acc[s0].items()}
    print(f'A_{s0}',json.dumps(res[f'A_{s0}'],ensure_ascii=False),flush=True)
def run_t(fname=None,side=None):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5) or t<6: continue
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
        selq&=np.nan_to_num(close_raw[t],nan=0)>=2.0
        nlv,_,_=feat5(t)
        vv=nlv[dom&~np.isnan(nlv)]
        if len(vv)>200:
            lo=np.nanquantile(vv,1/3)
            selq&=~(np.nan_to_num(nlv,nan=9)<lo)
        if fname is not None:
            v=dict(zip(FN,tfeat(t)))[fname]
            vv2=v[dom&~np.isnan(v)]
            if len(vv2)>200:
                lo2,hi2=np.nanquantile(vv2,[1/3,2/3])
                if side=='excl_low': selq&=~(np.nan_to_num(v,nan=9)<lo2)
                else: selq&=~(np.nan_to_num(v,nan=-9)>hi2)
        order=np.argsort(-np.nan_to_num(s,nan=-1e9))
        rank=np.full(N,1<<30); rank[order]=np.arange(N)
        can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
        new_h=dict(holdings)
        for i2 in list(new_h):
            if ((rank[i2]>=3.0*200) or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
        for i2 in order:
            if len(new_h)>=200: break
            if i2 in new_h or np.isnan(s[i2]) or not selq[i2]: continue
            new_h[i2]=0.0
        if len(new_h)<40: holdings={}; continue
        sc2=np.array([max(s[i2],0)+1.0 if s[i2]==s[i2] else 1.0 for i2 in new_h]); sc2/=sc2.sum()
        wt=dict(zip(new_h,sc2))
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
res['B_base']=ev(run_t(),'B_base')
for f in FN:
    for sd in ['excl_low','excl_high']:
        res[f'B_{f}_{sd}']=ev(run_t(f,sd),f'B_{f}_{sd}')
json.dump(res,open(f'{OUT}/metrics_v31_tick.json','w'),ensure_ascii=False,indent=1)
print('done')
