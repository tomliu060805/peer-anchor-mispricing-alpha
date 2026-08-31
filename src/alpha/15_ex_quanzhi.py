# -*- coding: utf-8 -*-
"""v11b: 定型空气指增(N200/H5/B3) 对中证全指(000985)的超额 + 净值图."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, INDEX_ROOT
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
CH=PROJ+'/charts/v11b_excess_vs_csiall'
os.makedirs(CH,exist_ok=True)
W,K,MOM,COST,WLIM,HOLD=120,5,20,0.0010,250,5
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
dnum=dates.astype('datetime64[D]')
hl=np.nan_to_num(at_hl,nan=0.0); chl=np.cumsum(hl,0)
own_lim=np.zeros((T,N),np.float32); own_lim[WLIM:]=chl[WLIM:]-chl[:-WLIM]
z=np.load(f'{CACHE}/nets_v8.npz'); rb=z['rebuilds']; NB_P,WV_P=z['nb_p'],z['wv_p']

# 000985 收益
def rd(i):
    f=f'{INDEX_ROOT}/price/price_daily/{dates[i]}.parquet'
    if not os.path.exists(f): return i,np.nan
    d=pd.read_parquet(f,columns=['code','close','pre_close'])
    r=d[d['code']=='000985.XSHG']
    if len(r)==0: return i,np.nan
    return i,float(r['close'].iloc[0])/float(r['pre_close'].iloc[0])-1
with ProcessPoolExecutor(50) as ex:
    outs=list(ex.map(rd,range(T),chunksize=20))
qz=np.zeros(T)
for i,v in outs:
    if v==v: qz[i]=v
lgq=np.cumsum(np.log1p(qz))
print('quanzhi loaded',flush=True)

sig_days=np.arange(rb[0]+1,T-1-HOLD,HOLD)
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
print('signal ready',flush=True)

NHOLD,BM=200,3.0
holdings={}; recs=[]
for si,t in enumerate(sig_days):
    if dnum[t]>END: break
    s=C[si].copy()
    dom=(own_lim[t]>=2)&~np.isnan(ret[t])&(st[t]<0.5)
    s[~dom]=np.nan
    can_buy=(paused[t]<0.5)&(at_hl[t]<0.5)
    can_sell=(paused[t]<0.5)&(at_ll[t]<0.5)
    order=np.argsort(-np.nan_to_num(s,nan=-1e9))
    rank=np.full(N,1<<30); rank[order]=np.arange(N)
    new_h=dict(holdings)
    for i2 in list(new_h):
        if (rank[i2]>=BM*NHOLD or np.isnan(s[i2])) and can_sell[i2]: del new_h[i2]
    for i2 in order:
        if len(new_h)>=NHOLD: break
        if i2 in new_h or not can_buy[i2] or np.isnan(s[i2]): continue
        new_h[i2]=0.0
    wt={i2:1.0/len(new_h) for i2 in new_h}
    turn=sum(abs(wt.get(i2,0)-holdings.get(i2,0)) for i2 in set(wt)|set(holdings))
    pr=sum(w*(np.exp(logc[t+HOLD,i2]-logc[t,i2])-1) for i2,w in wt.items())-turn*COST
    bq=np.exp(lgq[t+HOLD]-lgq[t])-1
    recs.append((dnum[t],pr,bq,turn/2))
    holdings=wt
ds=np.array([r[0] for r in recs]); pr=np.array([r[1] for r in recs]); bq=np.array([r[2] for r in recs])
ex=(1+pr)/(1+bq)-1
ann=252/HOLD
res={}
for segn,(a,b) in [('dev',DEV),('val',VAL),('full',('2015-01-01','2024-08-16'))]:
    m=(ds>=np.datetime64(a))&(ds<=np.datetime64(b))
    e=ex[m]; nav=np.cumprod(1+e)
    res[segn]={'ExAnn':round(float(nav[-1]**(ann/len(e))-1),4),
      'IR':round(float(e.mean()/(e.std()+1e-12)*np.sqrt(ann)),2),
      'TE':round(float(e.std()*np.sqrt(ann)),4),
      'ExMDD':round(float((1-nav/np.maximum.accumulate(nav)).max()),4),
      'n':int(len(e)),'win':round(float((e>0).mean()),3),'avg_bp':round(float(e.mean()*1e4),1),
      'pf':round(float(e[e>0].sum()/max(-e[e<0].sum(),1e-9)),2)}
years=sorted(set(str(d)[:4] for d in ds))
res['yearly_ex']={y:round(float(np.prod(1+ex[[str(d)[:4]==y for d in ds]])-1),4) for y in years}
json.dump(res,open(f'{OUT}/metrics_v11b_quanzhi.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False),flush=True)
np.savez(f'{OUT}/port_air_N200B3.npz',dates=ds.astype('datetime64[D]').astype(str),port=pr,quanzhi=bq,ex=ex)

fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(ds,np.cumprod(1+ex),lw=1.6,color='#d62728',label='空气指增 N200/B3 对中证全指 超额净值(费后)')
ax.plot(ds,np.cumprod(1+pr),lw=1.0,color='#1f77b4',alpha=.7,label='组合绝对净值')
ax.plot(ds,np.cumprod(1+bq),lw=1.0,color='#7f7f7f',ls='--',alpha=.7,label='中证全指(000985)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.set_yscale('log'); ax.set_title('活跃域联动锚多头(200只/周调/缓冲带600) vs 中证全指')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); v=[res['yearly_ex'][y] for y in years]
ax.bar(x,v,color=['#d62728' if q>=0 else '#2ca02c' for q in v])
for i,q in enumerate(v): ax.text(i,q+(0.005 if q>=0 else -0.02),f'{q*100:.1f}',ha='center',fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('对全指逐年超额(%, 复利, 费后)'); ax.grid(alpha=.3,axis='y')
plt.tight_layout(); plt.savefig(f'{CH}/excess_vs_csiall.png',dpi=130); print('chart saved')
