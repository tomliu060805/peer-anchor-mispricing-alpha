"""★收口检验 @双边20bp (训/验决定, 测试段封存):
  B1 永远做空基准(严格基准): 在【同样的突破时点】入场但方向恒为空 —— 若与策略相当, 则方向判定无价值
  B2 无条件做空基准: 每个决策点都做空(不要求突破) —— 检验"突破筛选"本身有无价值
  B3 ★组合层增量: 已有 v11.9 的前提下, 本策略(只空头腿)还剩多少边际
       A=v11.9超额 | B=本策略 | A+B | A+B(仅非red日) —— 后者是最纯的增量口径
"""
import os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); DEC=[1029,1129,1359]; CLOSE_HM=1457; NDAY=14; DPY=244
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
KEY={'CSI500':'500','CSI1000':'1000','CNI2000':'2000'}
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

def sim(code,k,mode='signal',maxtr=3):
    """mode: signal=按突破方向(仅空头腿) | always_short_on_break=突破时恒做空 | always_short=每决策点必做空"""
    g=_df[_df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC if T in pc.columns}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    nxt=dopen.shift(-1).values; fee=10e-4
    rows=[]
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]; ntr=0
        for T in DEC:
            if ntr>=maxtr or T not in sig or T not in hp: continue
            s_=sig[T].iloc[di]
            if not np.isfinite(s_): continue
            iT=hp[T]; cT=rc[iT]
            if not np.isfinite(cT): continue
            up,dn=o0*(1+k*s_), o0*(1-k*s_)
            brk = 1 if cT>up else (-1 if cT<dn else 0)
            if mode=='signal':
                if brk!=-1: continue
                side=-1
            elif mode=='always_short_on_break':
                if brk==0: continue
                side=-1
            else:
                side=-1
            j0=iT+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            if not np.isfinite(ep): continue
            xp=nxt[di]
            if not np.isfinite(xp): xp=rc[ic]
            rows.append((dstr,side*(xp/ep-1)-2*fee)); ntr+=1
    d=pd.DataFrame(rows,columns=['date','r']); d['dt']=pd.to_datetime(d['date'])
    return d

def rep(d,lab):
    o=[]
    for s,m in [('训',(d.dt>=TR0)&(d.dt<TRm)),('验',(d.dt>=TRm)&(d.dt<=END))]:
        t=d[m]['r'].values
        if not len(t): o.append(f'{s} 无'); continue
        w=t>0; pfn=-t[~w].sum()
        o.append(f'{s} {len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp '
                 f'PF{(t[w].sum()/pfn if pfn>0 else np.inf):>4.2f} 合计{t.sum()*100:>+7.1f}%')
    return f'  {lab:<34}' + ' | '.join(o)

CODES=['CSI500','CSI1000','CNI2000']
P('='*150); P('★ 收口检验 @双边20bp (净值; 训/验决定, 测试段2024-08起封存)')
P(); P('【B1/B2 严格基准: 方向判定与突破筛选各自值多少】')
for c in CODES:
    P(f'  ── {c}')
    for k in [2.0,2.5]:
        P(rep(sim(c,k,'signal'),f'k={k} 策略(只空头腿)'))
        P(rep(sim(c,k,'always_short_on_break'),f'k={k} 突破时恒做空'))
    P(rep(sim(c,2.0,'always_short'),'无条件做空(每决策点)'))

P(); P('【B3 ★组合层增量: 已有 v11.9, 本策略还剩多少】')
for c in CODES:
    sr=pd.read_parquet(f'{EXT_REGIME}/output/series_{KEY[c]}.parquet')
    sr['date']=pd.to_datetime(sr['date']); sr=sr.set_index('date')
    idx=sr.index[(sr.index>=TR0)&(sr.index<=END)]
    A=(sr['strat']-sr['bench']).reindex(idx).fillna(0)
    red=set(sr[sr['red']>0].index)
    P(f'  ── {c}')
    for k in [2.0,2.5]:
        d=sim(c,k,'signal')
        B=d.groupby('dt')['r'].sum().reindex(idx).fillna(0)
        dn=d[~d.dt.isin(red)]
        Bn=dn.groupby('dt')['r'].sum().reindex(idx).fillna(0)
        for lab,x in [('v11.9 单独',A),(f'k={k} 本策略单独',B),(f'k={k} v11.9+本策略',A+B),
                      (f'k={k} v11.9+本策略(仅非red日)',A+Bn)]:
            o=[]
            for s,m in [('训',(idx>=TR0)&(idx<TRm)),('验',(idx>=TRm)&(idx<=END))]:
                xx=x[m]; ann=xx.mean()*DPY*100; vol=xx.std()*np.sqrt(DPY)*100
                o.append(f'{s} 年化{ann:>+6.2f}% SR{(ann/vol if vol>0 else np.nan):>5.2f}')
            o.append(f'与v11.9相关{x.corr(A):>+5.2f}')
            P(f'    {lab:<30}' + ' | '.join(o))
        if k==2.0: P()
open(os.path.join(PROJ,'output','beta','final_controls.txt'),'w').write('\n'.join(lines))
print('\nsaved output/final_controls.txt')
