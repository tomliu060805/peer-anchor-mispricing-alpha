"""★当前定版形态的净值与逐年 (k=2.5, 4时点10:00/10:29/11:29/13:59, 只空头, 持到次日开盘, 双边20bp)。
★仓位口径: 同日多笔会重叠(都持到次日开盘) ⇒ 净值按【单位资金等分给当日信号】计,
   即 日收益 = 当日各笔收益的均值。同时报告同日笔数分布, 以量化重叠程度。
★测试段 2024-08-01 起封存, 不计算不作图。
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); CLOSE_HM=1457; NDAY=14; FEE=10e-4; K=2.5
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15'); DEC=[1000,1029,1129,1359]
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

def run(code):
    g=_df[_df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC if T in pc.columns}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]; nxt=dopen.shift(-1).values
    per_day={}; trades=[]
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]; day=[]
        for T in DEC:
            if T not in sig or T not in hp: continue
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT>=o0*(1-K*s_): continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            if not np.isfinite(ep): continue
            xp=nxt[di] if np.isfinite(nxt[di]) else rc[ic]
            if not np.isfinite(xp): continue
            r=-(xp/ep-1)-2*FEE; day.append(r); trades.append(r)
        per_day[pd.Timestamp(dstr)]=(np.mean(day),np.sum(day)) if day else (0.0,0.0)
    s=pd.DataFrame(per_day,index=['eq','sum']).T.sort_index()
    return s, np.array(trades), pc.index

CODES=[('CSI500','tab:blue'),('CSI1000','tab:orange'),('CNI2000','tab:green')]
fig,axes=plt.subplots(2,1,figsize=(13,8),gridspec_kw={'height_ratios':[2,1]})
YR={}
P('★ 定版形态: k=2.5, 4时点(10:00/10:29/11:29/13:59), 只空头, 持到次日开盘, 双边20bp')
P('  仓位口径: 单位资金等分给当日信号(日收益=当日各笔均值); 测试段2024-08起封存')
for code,col in CODES:
    S,tr,_=run(code)
    W=(S.index>=TR0)&(S.index<=END)
    S=S[W]; r=S['sum']; req=S['eq']
    axes[0].plot(S.index,r.cumsum()*100,lw=1.4,color=col,label=f'{code} 每信号一单位')
    axes[0].plot(S.index,req.cumsum()*100,lw=1.0,ls='--',alpha=.7,color=col,label=f'{code} 等分资金')
    YR[code]=r.groupby(r.index.year).sum()*100
    ann=r.mean()*244*100; vol=r.std()*np.sqrt(244)*100
    eq=(1+r).cumprod(); mdd=((eq/eq.cummax())-1).min()*100
    ndays=(r!=0).sum(); ann_eq=req.mean()*244*100
    tr_m=(r.index>=TR0)&(r.index<TRm); va_m=(r.index>=TRm)&(r.index<=END)
    P(f'  {code:<9} 全段 年化{ann:>+6.2f}% 波动{vol:>5.2f}% Sharpe{ann/vol:>5.2f} MDD{mdd:>7.2f}% '
      f'有仓日{ndays:>4}({ndays/len(r)*100:.0f}%) | 训 年化{r[tr_m].mean()*244*100:>+6.2f}% '
      f'验 年化{r[va_m].mean()*244*100:>+6.2f}% | 等分口径年化{ann_eq:>+5.2f}% | {len(tr)}笔 胜{(tr>0).mean()*100:.0f}% 单笔{tr.mean()*1e4:+.1f}bp')
axes[0].axvline(TRm,color='k',lw=0.8,ls=':')
axes[0].text(TRm,axes[0].get_ylim()[1]*0.95,' 训|验',fontsize=9,va='top')
axes[0].set_title('早盘噪声突破·空头腿(k=2.5, 4时点, 持隔夜, 双边20bp) 累计收益%  实线=每信号一单位(最高4x) 虚线=等分资金(1x)  测试段封存')
axes[0].grid(alpha=0.3); axes[0].legend()
yrs=sorted(set().union(*[set(v.index) for v in YR.values()]))
x=np.arange(len(yrs)); w=0.26
for i,(code,col) in enumerate(CODES):
    v=[YR[code].get(y,0) for y in yrs]
    b=axes[1].bar(x+(i-1)*w,v,w,color=col,label=code)
    for bb,vv in zip(b,v): axes[1].text(bb.get_x()+bb.get_width()/2,vv+(0.3 if vv>=0 else -1.2),f'{vv:+.0f}',ha='center',fontsize=7)
axes[1].axhline(0,color='k',lw=0.7); axes[1].set_xticks(x); axes[1].set_xticklabels([str(y) for y in yrs])
axes[1].set_title('逐年收益% (2024=截至07-31)'); axes[1].grid(alpha=0.3,axis='y'); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PROJ,'charts','beta','current_version_nav.png'),dpi=140)
P(); P('【逐年收益%】')
for code,_ in CODES:
    P(f'  {code:<9}' + ' '.join(f'{y}:{YR[code].get(y,0):+6.1f}' for y in yrs))
open(os.path.join(PROJ,'output','beta','plot_current.txt'),'w').write('\n'.join(lines))
print('saved figs/current_version_nav.png')
