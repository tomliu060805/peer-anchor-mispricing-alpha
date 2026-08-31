"""定版 S2 口径净值与逐年图。测试段已于2026-08-28开封, 以醒目分界标出。"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
CLOSE_HM=1457; NDAY=14; FEE=10e-4; K=2.5; DEC=[1000,1029,1129,1359]; UNIT=0.25
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
VAL_END=pd.Timestamp('2024-07-31'); TE0=pd.Timestamp('2024-08-01')
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet')); df['dt']=pd.to_datetime(df['date'])
def run(code):
    g=df[df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]; nxt=dopen.shift(-1).values
    rows=[]
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]
        for T in DEC:
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT>=o0*(1-K*s_): continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]; xp=nxt[di] if np.isfinite(nxt[di]) else rc[ic]
            if not (np.isfinite(ep) and np.isfinite(xp)): continue
            rows.append((pd.Timestamp(dstr),-(xp/ep-1)-2*FEE))
    d=pd.DataFrame(rows,columns=['dt','r'])
    return (d.groupby('dt')['r'].sum()*UNIT).reindex(pd.to_datetime(pc.index)).fillna(0)
fig,axes=plt.subplots(2,1,figsize=(13.5,8),gridspec_kw={'height_ratios':[2,1]})
YR={}; COL={'CSI500':'tab:blue','CSI1000':'tab:orange','CNI2000':'tab:green'}
for code,col in COL.items():
    r=run(code); r=r[r.index>=TR0]
    axes[0].plot(r.index,r.cumsum()*100,lw=1.4,color=col,label=code)
    YR[code]=r.groupby(r.index.year).sum()*100
axes[0].axvline(TRm,color='k',lw=0.9,ls=':'); axes[0].axvline(TE0,color='crimson',lw=1.1,ls='--')
yl=axes[0].get_ylim()
axes[0].text(TRm,yl[1]*0.96,' 训|验',fontsize=9,va='top')
axes[0].text(TE0,yl[1]*0.96,' 验|测(2024-08开封)',fontsize=9,va='top',color='crimson')
axes[0].axvspan(TE0,r.index.max(),color='crimson',alpha=0.05)
axes[0].set_title('早盘噪声突破·空头腿 v1 (S2口径: 每笔0.25单位/峰值1x, k=2.5, 4时点, 持隔夜, 双边20bp) 累计收益%')
axes[0].grid(alpha=0.3); axes[0].legend()
yrs=sorted(set().union(*[set(v.index) for v in YR.values()])); x=np.arange(len(yrs)); w=0.26
for i,(code,col) in enumerate(COL.items()):
    v=[YR[code].get(y,0) for y in yrs]
    b=axes[1].bar(x+(i-1)*w,v,w,color=col,label=code)
    for bb,vv in zip(b,v): axes[1].text(bb.get_x()+bb.get_width()/2,vv+(0.12 if vv>=0 else -0.45),f'{vv:+.1f}',ha='center',fontsize=7)
for i,y in enumerate(yrs):
    if y>=2024: axes[1].axvspan(i-0.5,i+0.5,color='crimson',alpha=0.05)
axes[1].axhline(0,color='k',lw=0.7); axes[1].set_xticks(x); axes[1].set_xticklabels([str(y) for y in yrs])
axes[1].set_title('逐年收益% (2024后含测试段, 淡红底)'); axes[1].grid(alpha=0.3,axis='y'); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PROJ,'charts','beta','s2_nav_yearly.png'),dpi=140)
for c,v in YR.items(): print(f'{c:<9}'+' '.join(f'{y}:{x_:+.1f}' for y,x_ in v.items()))
print('saved figs/s2_nav_yearly.png')
