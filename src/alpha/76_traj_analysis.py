# -*- coding: utf-8 -*-
"""v44(方向1): 执行轨迹特征分析. 六个新特征各自的条件表 + 产品变体, 与现行sweep/osize对照.
特征(卖方为主, 均前5日聚合/总成交额归一):
 lvl3_s 穿透>=3价位金额占比 | fast_s 母单<=1秒金额占比 | slow_s 母单>=60秒占比
 split_s 算法拆单占比(笔数>=5且单笔<5万) | durw_s 金额加权时长 | pcw_s 金额加权笔数
 imb_lvl 买卖穿透失衡"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
src=open(PROJ+'/src/alpha/68_behav_decay.py').read()
head=src.split("\nres={}\n")[0]
exec(head)
code_ix={c:i for i,c in enumerate(codes)}
TJ=PROJ+'/cache/ticktraj'
COLS=['total_money','as_n','as_amt','as_sweep','as_lvl3','as_fast','as_slow','as_split','as_durw','as_pcw',
      'ab_n','ab_amt','ab_sweep','ab_lvl3','ab_fast','ab_slow','ab_split','ab_durw','ab_pcw']
def load_tj(i):
    ds=str(dates[i]); f=f'{TJ}/{ds}.parquet'
    out=np.full((len(COLS),N),np.nan,np.float32)
    if not os.path.exists(f): return i,out
    d=pd.read_parquet(f)
    suf=np.where(d['exchange'].values=='SZ','.XSHE','.XSHG')
    full=np.char.add(d['code'].values.astype(str),suf)
    ix=np.array([code_ix.get(c,-1) for c in full]); okm=ix>=0
    for j,col in enumerate(COLS):
        if col in d.columns:
            v=pd.to_numeric(d[col],errors='coerce').values.astype(np.float32)
            out[j,ix[okm]]=np.nan_to_num(v[okm],nan=0.0)
    return i,out
if not os.path.exists(f'{CACHE}/tj_grid.npz'):
    with ProcessPoolExecutor(40) as ex:
        outs=list(ex.map(load_tj,range(T),chunksize=8))
    TJG=np.full((T,len(COLS),N),np.nan,np.float32)
    for i,o in outs: TJG[i]=o
    np.savez_compressed(f'{CACHE}/tj_grid.npz',tj=TJG)
    print('tj grid cached',flush=True)
TJG=np.load(f'{CACHE}/tj_grid.npz')['tj']
tjc=np.nancumsum(np.nan_to_num(TJG,nan=0.0),axis=0)
IDX={c:i for i,c in enumerate(COLS)}
def tj_feat(t,lag=0):
    e=t-lag; s=tjc[e]-tjc[e-5]
    tm=np.maximum(s[IDX['total_money']],1.0)
    f={}
    f['lvl3_s']=s[IDX['as_lvl3']]/tm
    f['fast_s']=s[IDX['as_fast']]/tm
    f['slow_s']=s[IDX['as_slow']]/tm
    f['split_s']=s[IDX['as_split']]/tm
    f['durw_s']=s[IDX['as_durw']]/np.maximum(s[IDX['as_n']],1.0)
    f['pcw_s']=s[IDX['as_pcw']]/np.maximum(s[IDX['as_n']],1.0)
    f['imb_lvl']=(s[IDX['ab_lvl3']]-s[IDX['as_lvl3']])/tm
    f['imb_fast']=(s[IDX['ab_fast']]-s[IDX['as_fast']])/tm
    has=(tjc[e,IDX['total_money']]-tjc[e-5,IDX['total_money']])>0
    return {k:np.where(has,v,np.nan).astype(np.float32) for k,v in f.items()}
TFN=['lvl3_s','fast_s','slow_s','split_s','durw_s','pcw_s','imb_lvl','imb_fast']
def seg_of(t):
    if dnum[t]<np.datetime64('2016-01-01'): return None
    if dnum[t]<=np.datetime64(DEV[1]): return 'dev'
    if dnum[t]<=np.datetime64(VAL[1]): return 'val'
    return None
def tstat(a): a=np.asarray(a,float); a=a[~np.isnan(a)]; return float(a.mean()/(a.std()+1e-12)*np.sqrt(max(len(a),1)))
res={}
acc={s:{f:{q:[] for q in range(3)} for f in TFN} for s in ['dev','val']}
for t in sig5:
    s0=seg_of(t)
    if s0 is None or t<8: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    a,b2=rank_xs(PG_b[t]/np.maximum(vol20[t]*np.sqrt(20),1e-4)),rank_xs(ZS_b[t])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        sc=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))[ix]
    fm=tj_feat(t)
    fwd=np.exp(logc[min(t+5,T-1),ix]-logc[t,ix])-1
    fu=np.nanmean(fwd)
    thr=np.nanquantile(sc,0.9); top=sc>=thr
    for fname in TFN:
        v=fm[fname][ix]; m2=top&~np.isnan(v)
        if m2.sum()<20: continue
        qe=np.nanquantile(v[m2],[1/3,2/3]); qq=np.digitize(v[m2],qe)
        for q in range(3):
            mm=qq==q
            if mm.sum()>=5: acc[s0][fname][q].append(float(np.nanmean(fwd[m2][mm])-fu))
for s0 in ['dev','val']:
    res[f'cond_{s0}']={f:{f'q{q}':{'bp':round(float(np.mean(v))*1e4,1),'t':round(tstat(v),2)} for q,v in d.items() if len(v)>20}
                       for f,d in acc[s0].items()}
    print(f'cond_{s0}',json.dumps(res[f'cond_{s0}'],ensure_ascii=False),flush=True)
json.dump(res,open(f'{OUT}/metrics_v44_traj.json','w'),ensure_ascii=False,indent=1)
print('cond done')
