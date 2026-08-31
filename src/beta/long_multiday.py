"""★日内做多机会检验: 核心假设——做多在20bp下必须【多日持有】摊薄费用。
理由: 日内漂移 +6~10bp/天 < 往返20bp; 无条件日内多头 20bp 下 −19~−29%/年。
两个候选(均为日内触发, 但持有N天):
  L1 突破做多: close(T)>Upper(T) → 次bar开盘买入 → 持N个交易日, 第N日14:57平
  L2 隔夜反转做多: 大幅低开(gap 在过去60日分位<q) → 09:31开盘买入 → 持N日
     (Zhang-Zhang-Xue 2025: A股隔夜收益负向预测首半小时, 机制=集合竞价+T+1)
N=0 表示当日14:57平(原版)。费用双边20bp, 每笔0.25单位(S2口径)。
★声明: 测试段(2024-08起)此前已开封, 本轮不再作为无偏证据, 仅列观察。
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
CLOSE_HM=1457; NDAY=14; FEE=10e-4; UNIT=0.25
DEC=[1000,1029,1129,1359]
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15'); TE0=pd.Timestamp('2024-08-01')
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet')); df['dt']=pd.to_datetime(df['date'])
lines=[]
def P(s=''):
    print(s); lines.append(s)

def prep(code):
    g=df[df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    return pc,po,hms,hp,po[hms[0]]

def L1(code,k,N):
    pc,po,hms,hp,dopen=prep(code)
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    close57=pc.iloc[:,ic].values; rows=[]
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]
        for T in DEC:
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT<=o0*(1+k*s_): continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            xi=di+N
            if xi>=len(pc.index): continue
            xp=close57[xi]
            if not (np.isfinite(ep) and np.isfinite(xp)): continue
            rows.append((pd.Timestamp(dstr),(xp/ep-1)-2*FEE))
    return pd.DataFrame(rows,columns=['dt','r'])

def L2(code,q,N):
    pc,po,hms,hp,dopen=prep(code)
    ic=np.where(hms<=CLOSE_HM)[0][-1]
    prev=pc.iloc[:,-1].shift(1)
    gap=(dopen/prev-1)
    pct=gap.rolling(60,min_periods=40).apply(lambda w:(w[:-1]<w[-1]).mean(),raw=True)
    close57=pc.iloc[:,ic].values; op=dopen.values; rows=[]
    for di,dstr in enumerate(pc.index):
        p=pct.iloc[di]
        if not np.isfinite(p) or p>=q: continue
        ep=op[di]; xi=di+N
        if xi>=len(pc.index): continue
        xp=close57[xi]
        if not (np.isfinite(ep) and np.isfinite(xp)): continue
        rows.append((pd.Timestamp(dstr),(xp/ep-1)-2*FEE))
    return pd.DataFrame(rows,columns=['dt','r'])

def rep(d,lab):
    o=[]
    for nm,a,b in [('训',TR0,TRm),('验',TRm,TE0),('测*',TE0,pd.Timestamp('2100-01-01'))]:
        t=d[(d.dt>=a)&(d.dt<b)]['r'].values
        if not len(t): o.append(f'{nm} 无'); continue
        w=t>0; pfn=-t[~w].sum()
        yrs=max((min(b,d.dt.max())-a).days/365.25,0.1)
        o.append(f'{nm} 年化{t.sum()*UNIT/yrs*100:>+6.2f}% {len(t):>4}笔 胜{w.mean()*100:>3.0f}% '
                 f'单笔{t.mean()*1e4:>+6.1f}bp PF{(t[w].sum()/pfn if pfn>0 else np.inf):>4.2f}')
    return f'  {lab:<26}' + ' | '.join(o)

P('='*150); P('★ 日内做多机会: 多日持有摊薄费用 (双边20bp, S2口径每笔0.25单位)')
P('  基线事实: 日内漂移 CSI500 +6.2 / CSI1000 +7.2 / CNI2000 +10.1 bp/日, 而往返20bp ⇒ 需持有2~3日才覆盖')
for code in ['CSI500','CSI1000','CNI2000']:
    P(); P(f'── {code}  【L1 突破做多 k=2.0】')
    for N in [0,1,2,3,5,10]:
        P(rep(L1(code,2.0,N),f'持有{N}日' if N else '当日14:57平(原版)'))
    P(f'  【L2 隔夜大幅低开做多 q=0.2】')
    for N in [0,1,2,3,5,10]:
        P(rep(L2(code,0.2,N),f'持有{N}日' if N else '当日14:57平'))
open(os.path.join(PROJ,'output','beta','long_multiday.txt'),'w').write('\n'.join(lines))
print('saved')
