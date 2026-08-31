# -*- coding: utf-8 -*-
"""融合超额曲线图: alpha单腿 vs 融合(h=1.0/1.5) 对中证全指"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R=PROJ+''
z=np.load(f'{R}/output/fusion/fusion_daily.npz')
ds=z['dates'].astype('datetime64[D]'); apr=z['alpha']; bench=z['bench']
# 口径铁律: 2015年剔出一切回测
_m=ds>=np.datetime64('2016-01-01'); ds,apr,bench=ds[_m],apr[_m],bench[_m]
df=pd.read_parquet(f'{R}/data/idx1m.parquet')
def beta_daily(code,K=2.5,NDAY=14,FEE=10e-4,DEC=(1000,1029,1129,1359),UNIT=0.25):
    g=df[df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={Td:(pc[Td]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for Td in DEC}
    C,O=pc.values,po.values; nxt=dopen.shift(-1).values
    dts=pd.to_datetime(pc.index); pnl={}
    for di in range(len(pc.index)-1):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        for Td in DEC:
            s=sig[Td].iloc[di]
            if not np.isfinite(s): continue
            j=hp.get(Td)
            if j is None or j+1>=len(hms): continue
            if not np.isfinite(C[di,j]) or C[di,j]>=o0*(1-K*s): continue
            e,x=O[di,j+1],nxt[di]
            if not (np.isfinite(e) and np.isfinite(x)): continue
            pnl[dts[di+1]]=pnl.get(dts[di+1],0.0)+UNIT*((e/x-1)-FEE*2)
    out=np.zeros(len(ds)); m={pd.Timestamp(d):i for i,d in enumerate(ds)}
    for k,v in pnl.items():
        i=m.get(pd.Timestamp(k))
        if i is not None: out[i]=v
    return out
bd=beta_daily('CNI2000')
def ex(r): return (1+r)/(1+bench)-1
series={'alpha单腿 v27':ex(apr),'融合 h=1.0(定案)':ex(apr+bd)}
ann=252
fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
cols={'alpha单腿 v27':'#7f7f7f','融合 h=1.0(定案)':'#d62728'}
stat={}
for k,e in series.items():
    nav=np.cumprod(1+e)
    ax.plot(ds,nav,lw=1.6,color=cols[k],
            label=f'{k}: 年化超额{nav[-1]**(ann/len(e))-1:.1%} IR{e.mean()/e.std()*np.sqrt(ann):.2f} 超额MDD{(1-nav/np.maximum.accumulate(nav)).max():.1%}')
    stat[k]={'ExAnn':float(nav[-1]**(ann/len(e))-1),'IR':float(e.mean()/e.std()*np.sqrt(ann)),
             'ExMDD':float((1-nav/np.maximum.accumulate(nav)).max())}
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.set_yscale('log')
tk=[1,2,4,8,16,32]
ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_minor_locator(FixedLocator([]))
fm=ScalarFormatter(); fm.set_scientific(False); ax.yaxis.set_major_formatter(fm)
ax.set_ylabel('超额净值'); ax.set_title('alpha × beta 融合: 对中证全指超额净值(日频, 费后双边20bp)')
ax.legend(fontsize=9); ax.grid(alpha=.3)
ax=axes[1]
years=sorted(set(str(d)[:4] for d in ds)); x=np.arange(len(years)); w=0.38
for i,(k,e) in enumerate(series.items()):
    v=[float(np.prod(1+e[[str(d)[:4]==y for d in ds]])-1) for y in years]
    ax.bar(x+(i-0.5)*w,v,w,color=cols[k],label=k)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.legend(fontsize=9); ax.grid(alpha=.3,axis='y')
plt.tight_layout()
os.makedirs(f'{R}/charts/fusion',exist_ok=True)
plt.savefig(f'{R}/charts/fusion/fusion_excess_nav.png',dpi=130)
print(json.dumps(stat,ensure_ascii=False,indent=1)); print('saved')
