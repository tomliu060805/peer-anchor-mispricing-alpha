# -*- coding: utf-8 -*-
"""v30c: 元指标产品化. 三个仪表周频序列+图:
1) 拥挤度: 滚动52周 多头腿h1超额/h20超额占比 (>40%预警)
2) regime: 滚动26周 同行业锚-隐性锚 多头腿超额差 (正=行业市, 负=题材市)
3) 网络换血率: 域内均值churn (来自v30b缓存)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
src=open(PROJ+'/src/alpha/48_combo_final.py').read()
head=src.split("res={}")[0]
exec(head)
nz=np.load(f'{CACHE}/nets_v28_ind.npz')
z8=np.load(f'{CACHE}/nets_v8.npz'); RB8=z8['rebuilds']
def pg_of(NB,WV):
    PG=np.full((T,N),np.nan,np.float32)
    for t in sig5:
        b=np.searchsorted(RB8,t,side='right')-1
        nbr=NB[b]; wgt=np.maximum(WV[b],0)
        idx=np.where((nbr[:,0]>=0)&(wgt.sum(1)>1e-9))[0]
        nb=nbr[idx]; w0=wgt[idx]
        m=mom20[t]; pm=m[np.where(nb>=0,nb,0)]
        msk=(nb>=0)&~np.isnan(pm); w=w0*msk
        sw=np.maximum(w.sum(1),1e-9)
        mu=(np.nan_to_num(pm)*w).sum(1)/sw
        ok=w.sum(1)>1e-9
        PG[t,idx[ok]]=mu[ok]-m[idx[ok]]
    return PG
PG_cr=pg_of(nz['cross_n'],nz['cross_w']); PG_sa=pg_of(nz['same_n'],nz['same_w'])
rows=[]
for t in sig5:
    if t+21>=T: continue
    dom=(lim250[t]>=2)&tradable[t]&~np.isnan(PG_b[t])
    ix=np.where(dom)[0]
    if len(ix)<400: continue
    def longleg_ex(PGx,h):
        gv=PGx[t,ix]
        m2=~np.isnan(gv)
        if m2.sum()<200: return np.nan
        sc=gv[m2]/np.maximum(vol20[t,ix][m2]*np.sqrt(20),1e-4)
        thr=np.quantile(sc,0.9); top=ix[m2][sc>=thr]
        return float(np.nanmean(np.exp(logc[t+h,top]-logc[t,top])-1)-np.nanmean(np.exp(logc[t+h,ix]-logc[t,ix])-1))
    rows.append((dnum[t],longleg_ex(PG_b,1),longleg_ex(PG_b,20),longleg_ex(PG_sa,5),longleg_ex(PG_cr,5)))
ds=np.array([r[0] for r in rows]); h1=np.array([r[1] for r in rows],float); h20=np.array([r[2] for r in rows],float)
sa=np.array([r[3] for r in rows],float); cr=np.array([r[4] for r in rows],float)
def roll(x,w):
    out=np.full(len(x),np.nan)
    for i in range(w-1,len(x)): out[i]=np.nanmean(x[max(0,i-w+1):i+1])
    return out
rh1=roll(h1,52); rh20=roll(h20,52)
# 稳健: h20滚动均值过小(<10bp)时比值无意义, 置NaN; 并截断到[-0.2,1.2]
crowd=np.where(rh20>0.0010, rh1/np.maximum(rh20,1e-9), np.nan)
crowd=np.clip(crowd,-0.2,1.2)
unstable=~(rh20>0.0010)
regime=roll(sa-cr,26)*1e4
chz=np.load(f'{OUT}/mkt_churn_series.npz')
chd=chz['dates'].astype('datetime64[D]'); chv=roll(chz['churn'],13)
np.savez(f'{OUT}/meta_gauges.npz',dates=ds.astype('datetime64[D]').astype(str),crowd=crowd,regime=regime)
fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
ax=axes[0]
ax.plot(ds,crowd,lw=1.4,color='#d62728')
for i in np.where(unstable)[0]:
    ax.axvspan(ds[max(i-1,0)],ds[min(i+1,len(ds)-1)],color='gray',alpha=.12,lw=0)
ax.axhline(0.40,color='k',ls='--',lw=.8); ax.text(ds[60],0.43,'拥挤预警线 40%',fontsize=9)
ax.set_title('仪表1: 拥挤度 = 滚动52周 h1超额/h20超额 (灰带=长端alpha过小比值无意义)'); ax.grid(alpha=.3)
ax.set_ylim(-0.2,1.2)
ax=axes[1]
ax.plot(ds,regime,lw=1.4,color='#1f77b4')
ax.axhline(0,color='k',lw=.6)
ax.fill_between(ds,0,regime,where=regime>0,alpha=.2,color='#1f77b4',label='行业市(同行业锚占优)')
ax.fill_between(ds,0,regime,where=regime<0,alpha=.2,color='#d62728',label='题材市(隐性锚占优)')
ax.set_title('仪表2: regime = 滚动26周 同行业锚-隐性锚 多头超额差(bp/周)'); ax.legend(fontsize=9); ax.grid(alpha=.3)
ax=axes[2]
ax.plot(chd,chv,lw=1.4,color='#2ca02c')
ax.set_title('仪表3: 网络换血率 = 域内top5邻居换血比例(滚动13周)'); ax.grid(alpha=.3)
plt.tight_layout(); 
CH=PROJ+'/charts/v30_meta_gauges'
os.makedirs(CH,exist_ok=True)
plt.savefig(f'{CH}/three_gauges.png',dpi=130)
print('latest values:',json.dumps({'crowd':round(float(crowd[-1]),3),'regime_bp':round(float(regime[-1]),1),'churn':round(float(chv[-1]),3)}))
print('chart saved')
