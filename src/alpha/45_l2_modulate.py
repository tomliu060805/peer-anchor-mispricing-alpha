# -*- coding: utf-8 -*-
"""v24c: L2资金流调制. 特征(信号日前5日聚合):
  nl5 = 大单净流入/成交额;  ab5 = 主动买占比;  rt5 = 小单净流入/成交额
A) 多头腿按特征三分位的fwd超额 (象限式条件表)
B) 产品变体: 买入剔除某特征最不利三分位 (方向由A定, 两个方向都跑)
覆盖: 2016(补算)~2024.08"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/32_retest_all.py').read()
head=src.split("def run(")[0]
exec(head)
code_ix={c:i for i,c in enumerate(codes)}
MF1=PROJ+'/cache/moneyflow_ext'
MF2=os.environ.get('MONEYFLOW_ROOT','')
def load_mf(i):
    ds=str(dates[i])
    f=f'{MF1}/{ds}.parquet' if os.path.exists(f'{MF1}/{ds}.parquet') else f'{MF2}/{ds}.parquet'
    out=np.full((6,N),np.nan,np.float32)  # large_buy,large_sell,total_buy,total_sell,total_money,net_large(含x? 用large即可)
    if not os.path.exists(f): return i,out
    d=pd.read_parquet(f,columns=['code','exchange','large_buy','large_sell','total_buy','total_sell','total_money'])
    suf=np.where(d['exchange'].values=='SZ','.XSHE','.XSHG')
    full=np.char.add(d['code'].values.astype(str),suf)
    ix=np.array([code_ix.get(c,-1) for c in full])
    okm=ix>=0
    vals=d[['large_buy','large_sell','total_buy','total_sell','total_money']].values.astype(np.float32).T
    for j in range(5): out[j,ix[okm]]=vals[j][okm]
    return i,out
if not os.path.exists(f'{CACHE}/mf_grid.npz'):
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(load_mf,range(T),chunksize=8))
    MF=np.full((T,6,N),np.nan,np.float32)
    for i,o in outs: MF[i]=o
    np.savez_compressed(f'{CACHE}/mf_grid.npz',mf=MF[:,:5])
    print('mf grid cached',flush=True)
MF=np.load(f'{CACHE}/mf_grid.npz')['mf']  # (T,5,N)
mfc=np.nancumsum(np.nan_to_num(MF,nan=0.0),axis=0)
def feat5(t):
    s=mfc[t]-mfc[t-5]   # 5日聚合 (t-4..t)... 注意实时性: 信号日t收盘后可得
    lb,ls,tb,ts,tm=s
    tm=np.maximum(tm,1.0)
    nl=(lb-ls)/tm
    ab=tb/np.maximum(tb+ts,1.0)
    rt=((tb-lb)-(ts-ls))/tm
    has=(mfc[t,4]-mfc[t-5,4])>0
    return np.where(has,nl,np.nan),np.where(has,ab,np.nan),np.where(has,rt,np.nan)
PG_b,PM_b,DSP_b,ZS_b=S['price']
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
res={}
# A) 条件表
acc={s0:{f:{q:[] for q in range(3)} for f in ['nl5','ab5','rt5']} for s0 in ['dev','val']}
for t in sig5:
    s0=seg_of(t)
    if s0 is None: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    gap=PG_b[t,ix]/np.maximum(vol20[t,ix]*np.sqrt(20),1e-4)
    a,b2=rank_xs(PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)),rank_xs(ZS_b[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        sc=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))[ix]
    f5=feat5(t)
    fwd=np.exp(logc[min(t+5,T-1),ix]-logc[t,ix])-1
    fu=np.nanmean(fwd)
    thr=np.nanquantile(sc,0.9); top=sc>=thr
    for fname,fv in zip(['nl5','ab5','rt5'],f5):
        v=fv[ix]
        m2=top&~np.isnan(v)
        if m2.sum()<20: continue
        qe=np.nanquantile(v[m2],[1/3,2/3])
        qq=np.digitize(v[m2],qe)
        for q in range(3):
            mm=qq==q
            if mm.sum()>=5: acc[s0][fname][q].append(float(np.nanmean(fwd[m2][mm])-fu))
for s0 in ['dev','val']:
    res[f'A_{s0}']={f:{f'q{q}':{'bp':round(float(np.mean(v))*1e4,1),'t':round(tstat(v),2)}
                      for q,v in d.items() if len(v)>20} for f,d in acc[s0].items()}
    print(f'A_{s0}',json.dumps(res[f'A_{s0}']),flush=True)
# B) 产品变体
def run_l2(feat_name=None,side=None):
    holdings={}; recs=[]
    for si,t in enumerate(sig5):
        if si+1>=len(sig5): break
        gap=PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)
        a,b2=rank_xs(gap),rank_xs(ZS_b[t])
        stk=np.stack([a,b2])
        with np.errstate(all='ignore'):
            s=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
        dom=(lim250[t]>=0.5+1.5)&tradable[t]&~np.isnan(PG_b[t])
        s=np.where(dom,s,np.nan)
        dm=np.nanmean(np.where(dom,mom20[t],np.nan))
        selq=(paused[t]<0.5)&(at_hl[t]<0.5)&(hl5[t]<1)&(ll5[t]<1)
        selq&=~((PM_b[t]-dm>=0)&(mom20[t]-dm>=0))
        if feat_name is not None:
            fmap=dict(zip(['nl5','ab5','rt5'],feat5(t)))
            v=fmap[feat_name]
            vv=v[dom&~np.isnan(v)]
            if len(vv)>200:
                lo,hi=np.nanquantile(vv,[1/3,2/3])
                if side=='excl_low': selq&=~(np.nan_to_num(v,nan=9)<lo)
                else: selq&=~(np.nan_to_num(v,nan=-9)>hi)
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
res['B_base']=ev(run_l2(),'B_base')
for f in ['nl5','ab5','rt5']:
    for sd in ['excl_low','excl_high']:
        res[f'B_{f}_{sd}']=ev(run_l2(f,sd),f'B_{f}_{sd}')
json.dump(res,open(f'{OUT}/metrics_v24c_l2.json','w'),ensure_ascii=False,indent=1)
print('done')
