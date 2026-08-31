"""★逐年增量: 本策略(S2口径)叠加在 v11.9 空头腿之上, 每年多贡献多少。
三列口径:
  v11.9        = strat − bench (官方超额序列)
  本策略        = S2 日收益(每笔0.25单位, 峰值1x)
  ★纯增量      = 本策略【仅非red日】部分 —— v11.9 已覆盖 red 日, 这才是它带来的新东西
测试段(2024-08起)已于2026-08-28开封, 标★区分。
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
CLOSE_HM=1457; NDAY=14; FEE=10e-4; K=2.5; DEC=[1000,1029,1129,1359]; UNIT=0.25
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15'); TE0=pd.Timestamp('2024-08-01')
KEY={'CSI500':'500','CSI1000':'1000','CNI2000':'2000'}
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet')); df['dt']=pd.to_datetime(df['date'])
lines=[]
def P(s=''):
    print(s); lines.append(s)
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
    return pd.DataFrame(rows,columns=['dt','r'])
P('='*126); P('★ 本策略(S2, 峰值1x)对 v11.9 空头腿的逐年增量 (均已扣双边20bp)')
P('  ★纯增量 = 本策略仅非red日部分(red日 v11.9 已覆盖)')
for code,key in KEY.items():
    s=pd.read_parquet(f'{EXT_REGIME}/output/series_{key}.parquet')
    s['date']=pd.to_datetime(s['date']); s=s.set_index('date')
    idx=s.index[s.index>=TR0]
    A=(s['strat']-s['bench']).reindex(idx).fillna(0)
    red=set(s[s['red']>0].index)
    d=run(code)
    B=(d.groupby('dt')['r'].sum()*UNIT).reindex(idx).fillna(0)
    dn=d[~d.dt.isin(red)]
    Bn=(dn.groupby('dt')['r'].sum()*UNIT).reindex(idx).fillna(0)
    yA=A.groupby(A.index.year).sum()*100
    yB=B.groupby(B.index.year).sum()*100
    yBn=Bn.groupby(Bn.index.year).sum()*100
    P(); P(f'── {code}')
    P(f'  {"年":<7}{"v11.9":>9}{"本策略":>9}{"合计":>9}{"★纯增量(非red日)":>18}{"":>4}')
    for y in sorted(yA.index):
        mark='★' if y>=2024 else ' '
        P(f'  {y}{mark:<6}{yA.get(y,0):>+9.2f}{yB.get(y,0):>+9.2f}{yA.get(y,0)+yB.get(y,0):>+9.2f}{yBn.get(y,0):>+18.2f}')
    for lab,m in [('训',(idx>=TR0)&(idx<TRm)),('验',(idx>=TRm)&(idx<TE0)),('★测',(idx>=TE0))]:
        yrs=(idx[m].max()-idx[m].min()).days/365.25
        P(f'  {lab:<6} 年化: v11.9{A[m].sum()/yrs*100:>+7.2f}%  本策略{B[m].sum()/yrs*100:>+7.2f}%  '
          f'合计{(A[m]+B[m]).sum()/yrs*100:>+7.2f}%  ★纯增量{Bn[m].sum()/yrs*100:>+7.2f}%  '
          f'相关{A[m].corr(B[m]):>+5.2f}')
open(os.path.join(PROJ,'output','beta','increment_vs_v119.txt'),'w').write('\n'.join(lines))
print('saved')
