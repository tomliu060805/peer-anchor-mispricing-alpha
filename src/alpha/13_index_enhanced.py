# -*- coding: utf-8 -*-
"""v10: 指增组合构建版.
信号 = Barra中性化合成秩 × 连续活跃度增益 g(act)=0.5+min(lim250,8)/8
组合 = 100只, 行业配比钉基准(csi_500 / csi_1000 权重文件×申万一级), 行业内等权
变体: A 行业匹配 B 行业匹配+剔除基准5%分位以下SIZE微盘尾
评估: 相对真实指数(000905/000852)收盘-收盘超额, 费=|Δw|×单边12.5bp, dev/val, 逐年
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, INDEX_ROOT, BARRA_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,K,MOM,HOLD,COST,WLIM=120,5,20,5,0.00125,250
NHOLD=100
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')
STYLES=['BETA','BTOP','EARNYILD','GROWTH','LEVERAGE','LIQUIDTY','MOMENTUM','RESVOL','SIZE','SIZENL']

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); chl=np.cumsum(hl,0)
own_lim=np.zeros((T,N),np.float32); own_lim[WLIM:]=chl[WLIM:]-chl[:-WLIM]
code_ix={c:i for i,c in enumerate(codes)}
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']
sig_days=np.arange(rb[0]+1,T-1-HOLD,HOLD)

# ---- 指数日收益 ----
idx_ret={}
for idxcode in ['000905.XSHG','000852.XSHG']:
    closes=np.full(T,np.nan)
    # 逐日文件读一次太慢, 批量并行
    def rd(i):
        f=f'{INDEX_ROOT}/price/price_daily/{dates[i]}.parquet'
        if not os.path.exists(f): return i,np.nan,np.nan
        d=pd.read_parquet(f,columns=['code','close','pre_close'])
        r=d[d['code']==idxcode]
        if len(r)==0: return i,np.nan,np.nan
        return i,float(r['close'].iloc[0]),float(r['pre_close'].iloc[0])
    with ProcessPoolExecutor(50) as ex:
        outs=list(ex.map(rd,range(T),chunksize=20))
    rr=np.full(T,np.nan)
    for i,c,pc in outs:
        if c==c and pc==pc and pc>0: rr[i]=c/pc-1
    idx_ret[idxcode]=np.nan_to_num(rr,nan=0.0)
    print(idxcode,'loaded',flush=True)
np.savez(f'{CACHE}/idx_ret.npz',**{k.replace('.','_'):v for k,v in idx_ret.items()})

# ---- 信号 (同v9): gap+zspread combo -> Barra中性化 ----
S_pg=np.full((len(sig_days),N),np.nan,np.float32)
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
S_zs=np.full((len(sig_days),N),np.nan,np.float32)
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
COMBO=np.full_like(S_pg,np.nan)
for si in range(len(sig_days)):
    a,b2=rank_xs(S_pg[si]),rank_xs(S_zs[si])
    stk=np.stack([a,b2])
    with np.errstate(all='ignore'):
        COMBO[si]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
print('signals ready',flush=True)

def load_expo(si):
    t=sig_days[si]; dprev=str(dates[t-1])
    out=np.full((N,len(STYLES)),np.nan,np.float32)
    for j,sty in enumerate(STYLES):
        f=f'{BARRA_ROOT}/{sty}/{dprev}.parquet'
        if not os.path.exists(f): return si,out
        d=pd.read_parquet(f)
        ix=np.array([code_ix.get(c,-1) for c in d['code']])
        v=d[sty].values.astype(np.float32)
        okm=ix>=0
        out[ix[okm],j]=v[okm]
    return si,out
with ProcessPoolExecutor(50) as ex:
    res_e=list(ex.map(load_expo,range(len(sig_days)),chunksize=4))
EXPO=np.full((len(sig_days),N,len(STYLES)),np.nan,np.float32)
for si,e in res_e: EXPO[si]=e
print('exposures loaded',flush=True)

def industry_arr(t):
    dstr=str(dates[t-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    return np.array([imap.get(c,'') for c in codes])

SNEU=np.full_like(COMBO,np.nan)
for si,t in enumerate(sig_days):
    s=rank_xs(COMBO[si]); E=EXPO[si]
    inds=industry_arr(t)
    m=~np.isnan(s)&~np.isnan(E).any(1)&tradable[t]&(inds!='')
    if m.sum()<300: continue
    uniq=[u for u in np.unique(inds[m])]
    D=np.zeros((m.sum(),len(uniq)),np.float32)
    imm=inds[m]
    for j,u in enumerate(uniq): D[imm==u,j]=1
    X=np.column_stack([np.ones(m.sum()),E[m],D])
    keep=X.std(0)>1e-9; keep[0]=True
    b2=np.linalg.lstsq(X[:,keep],s[m],rcond=None)[0]
    SNEU[si,m]=s[m]-X[:,keep]@b2
print('neutralized',flush=True)

# ---- 基准成分行业权重 (信号日) ----
def bench_weights(si,pool):
    t=sig_days[si]
    f=f'{INDEX_ROOT}/weight/{pool}/{dates[t]}.parquet'
    if not os.path.exists(f):
        # 回退找最近
        for lag in range(1,6):
            f=f'{INDEX_ROOT}/weight/{pool}/{dates[t-lag]}.parquet'
            if os.path.exists(f): break
    d=pd.read_parquet(f,columns=['stock_code','weight'])
    ix=np.array([code_ix.get(c,-1) for c in d['stock_code']])
    wv=d['weight'].values
    okm=ix>=0
    return ix[okm],wv[okm]

def build_port(pool,size_floor=False):
    """返回逐期 (date, port_ret_net, w_dict)"""
    act=lambda t: 0.5+np.minimum(own_lim[t],8)/8
    prev_w={}
    recs=[]
    for si,t in enumerate(sig_days):
        if dnum[t]>END: break
        s=SNEU[si].copy()
        gsc=act(t)
        score=s*gsc
        score[~tradable[t]]=np.nan
        bix,bw=bench_weights(si,pool)
        inds=industry_arr(t)
        # 基准行业权重
        bi=inds[bix]
        tot=bw.sum()
        ind_w={}
        for u in np.unique(bi):
            if u=='' : continue
            ind_w[u]=bw[bi==u].sum()/tot
        # SIZE 地板: 基准成分 SIZE 5%分位
        if size_floor:
            sz=EXPO[si][:,STYLES.index('SIZE')]
            floor=np.nanpercentile(sz[bix],5)
            score[np.nan_to_num(sz,nan=-9)<floor]=np.nan
        # 行业配额
        slots={u:ind_w[u]*NHOLD for u in ind_w}
        base_n={u:int(np.floor(v)) for u,v in slots.items()}
        rem=NHOLD-sum(base_n.values())
        fr=sorted(slots, key=lambda u: slots[u]-base_n[u], reverse=True)
        for u in fr[:rem]: base_n[u]+=1
        w_new={}
        for u,nsl in base_n.items():
            if nsl<=0: continue
            cand=np.where((inds==u)&~np.isnan(score))[0]
            if len(cand)==0: continue
            pick=cand[np.argsort(-score[cand])][:nsl]
            for i2 in pick: w_new[i2]=ind_w[u]/max(nsl,1)
        sw=sum(w_new.values())
        w_new={i2:w/sw for i2,w in w_new.items()}
        # 持有期收益 (t收盘->t+HOLD收盘)
        pr=0.0
        for i2,w in w_new.items():
            r5=np.exp(logc[t+HOLD,i2]-logc[t,i2])-1
            pr+=w*r5
        turn=0.0
        keys=set(w_new)|set(prev_w)
        for i2 in keys: turn+=abs(w_new.get(i2,0)-prev_w.get(i2,0))
        pr-=turn/2*2*COST   # 单边成本×双向变化量的一半×2 = turn×COST
        recs.append((dnum[t],pr,turn/2))
        prev_w=w_new
    return recs

def eval_port(recs,idxcode,name):
    res={}
    ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); tn=np.array([r[2] for r in recs])
    ir=idx_ret[idxcode]
    # 指数同窗收益
    tmap={d:i for i,d in enumerate(dnum)}
    br=[]
    lgi=np.cumsum(np.log1p(ir))
    for d in ds:
        t=tmap[d]
        br.append(np.exp(lgi[t+HOLD]-lgi[t])-1)
    br=np.array(br)
    ex=(1+pr)/(1+br)-1
    ann=252/HOLD
    for segn,(a,b) in [('dev',DEV),('val',VAL)]:
        m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
        e=ex[m]
        nav=np.cumprod(1+e)
        res[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),
          'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
          'TE':round(float(e.std()*np.sqrt(ann)),4),
          'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),
          'n':int(len(e)),'win':round(float((e>0).mean()),3),'avg_bp':round(float(e.mean()*1e4),1),
          'pf':round(float(e[e>0].sum()/max(-e[e<0].sum(),1e-9)),2),
          'turnover':round(float(tn[m].mean()),3)}
    years=sorted(set(str(d)[:4] for d in ds))
    res['yearly']={y:round(float(np.prod(1+ex[[str(d)[:4]==y for d in ds]])-1),4) for y in years}
    print(name,json.dumps(res),flush=True)
    return res,ds,ex

out={}
for pool,idxcode,tag in [('csi_500','000905.XSHG','500'),('csi_1000','000852.XSHG','1000')]:
    for sf in [False,True]:
        nm=f'enh{tag}{"_szfloor" if sf else ""}'
        recs=build_port(pool,size_floor=sf)
        r,ds,ex=eval_port(recs,idxcode,nm)
        out[nm]=r
        np.savez(f'{OUT}/port_{nm}.npz',dates=ds.astype('datetime64[D]').astype(str),ex=ex)
json.dump(out,open(f'{OUT}/metrics_v10_indexenh.json','w'),ensure_ascii=False,indent=1)
print('done')
