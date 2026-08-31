# -*- coding: utf-8 -*-
"""v5: 定型合成信号(peer_gap_K5+zspread)多头Top10% 费后超额净值 + 逐年超额.
基准=当期可交易全域等权均值; 超额按复利: (1+r_top_net)/(1+r_uni)-1."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
CH=PROJ+'/charts/v5_long_excess_nav'
os.makedirs(CH,exist_ok=True)
W,MOM,HOLD,COST=120,20,5,0.00125
END=np.datetime64('2024-08-16')   # test锁定

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
nw=np.load(f'{CACHE}/networks.npz'); rebuilds=nw['rebuilds']
nbr_resid,wgt_resid=nw['nbr_resid'],nw['wgt_resid']
z=np.load(f'{OUT}/signals_resid.npz'); sig_days=z['sig_days']; ZS=z['zspread']
def last_rebuild(t): return np.searchsorted(rebuilds,t,side='right')-1
fwd=np.full((T,N),np.nan,np.float32)
fwd[:T-HOLD]=(np.exp(logc[HOLD:]-logc[:-HOLD])-1).astype(np.float32)

PG5=np.full((len(sig_days),N),np.nan,np.float32)
for si,t in enumerate(sig_days):
    b=last_rebuild(t); nbr=nbr_resid[b][:,:5]; wgt=np.maximum(wgt_resid[b][:,:5],0)
    m=mom[t]; pm=np.take(m,np.where(nbr>=0,nbr,0))
    mask=(nbr>=0)&~np.isnan(pm); w=wgt*mask
    sw=w.sum(1); agg=(np.where(mask,np.nan_to_num(pm),0)*w).sum(1)/np.maximum(sw,1e-9)
    rows=np.where(sw>1e-9)[0]; PG5[si,rows]=agg[rows]-m[rows]
def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
def top_excess(S,label):
    ds=[];ex=[];prev=None
    for si,t in enumerate(sig_days):
        if dnum[t]>END or t+HOLD>=T: continue
        s=S[si].copy(); s[~tradable[t]]=np.nan; f=fwd[t]
        m=~np.isnan(s)&~np.isnan(f)
        if m.sum()<100: prev=None; continue
        sv,fv=s[m],f[m]; ids=np.where(m)[0]
        q=np.argsort(np.argsort(sv))/len(sv)
        top=set(ids[q>=0.9])
        to=1-len(top&prev)/max(len(top),1) if prev else 1.0
        rt=fv[q>=0.9].mean()-to*2*COST      # 费后多头
        ru=fv.mean()
        ds.append(dnum[t]); ex.append((1+rt)/(1+ru)-1)
        prev=top
    return np.array(ds),np.array(ex)

COMBO=np.full_like(PG5,np.nan)
for si in range(len(sig_days)):
    a,b2=rank_xs(PG5[si]),rank_xs(ZS[si])
    st2=np.stack([a,b2]); COMBO[si]=np.where(np.all(np.isnan(st2),0),np.nan,np.nanmean(st2,0))
S_rev=np.stack([-mom[t] for t in sig_days])

d1,e1=top_excess(COMBO,'combo'); d2,e2=top_excess(S_rev,'rev')
nav1=np.cumprod(1+e1); nav2=np.cumprod(1+e2)
years=sorted(set(str(d)[:4] for d in d1))
yr_ex={}
for y in years:
    m=np.array([str(d)[:4]==y for d in d1])
    m2=np.array([str(d)[:4]==y for d in d2])
    yr_ex[y]=(float(np.prod(1+e1[m])-1), float(np.prod(1+e2[m2])-1))
ann=252/HOLD
stats={'ann_excess':round(float(nav1[-1]**(ann/len(e1))-1),4),
 'sharpe':round(float(e1.mean()/e1.std()*np.sqrt(ann)),2),
 'n':int(len(e1)),'win':round(float((e1>0).mean()),3),'avg_bp':round(float(e1.mean()*1e4),1),
 'wl':round(float(e1[e1>0].mean()/-e1[e1<0].mean()),2),
 'pf':round(float(e1[e1>0].sum()/-e1[e1<0].sum()),2),
 'mdd':round(float((1-nav1/np.maximum.accumulate(nav1)).max()),4),
 'yearly':{y:[round(v[0],4),round(v[1],4)] for y,v in yr_ex.items()}}
json.dump(stats,open(f'{OUT}/metrics_v5_excess.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(stats,ensure_ascii=False))

fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(d1,nav1,lw=1.6,color='#d62728',label='联动合成 K5+zspread 多头Top10% 超额净值(费后)')
ax.plot(d2,nav2,lw=1.2,color='#7f7f7f',ls='--',label='纯反转 -mom20 多头Top10% 超额净值(费后)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.text(np.datetime64('2023-02-01'),nav1.max()*0.55,'val段起点',fontsize=9,color='gray')
ax.set_title('多头Top10% 相对全市场等权 复利超额净值 (周频, 单边12.5bp, 基准=当期可交易全域等权)')
ax.legend(); ax.grid(alpha=.3); ax.set_ylabel('超额净值')
ax=axes[1]
x=np.arange(len(years)); wd=0.38
c=[yr_ex[y][0] for y in years]; r=[yr_ex[y][1] for y in years]
ax.bar(x-wd/2,c,wd,color='#d62728',label='联动合成')
ax.bar(x+wd/2,r,wd,color='#7f7f7f',label='纯反转')
for i,v in enumerate(c): ax.text(i-wd/2,v+(0.004 if v>=0 else -0.012),f'{v*100:.1f}',ha='center',fontsize=8)
ax.axhline(0,color='k',lw=.6)
ax.set_xticks(x); ax.set_xticklabels(years); ax.set_title('逐年超额收益(%, 复利, 费后)')
ax.legend(); ax.grid(alpha=.3,axis='y')
plt.tight_layout(); plt.savefig(f'{CH}/long_excess_nav_and_yearly.png',dpi=130)
print('saved')
