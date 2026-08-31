"""★按入场时点拆解增量。用户观察: 13:59判定/14:00成交 与 v11.9 决策时刻重合 ⇒ 那一笔本质是v11.9重做。
真正可能有新信息的是 10:00 / 10:29 / 11:29 三笔。
每个时点单独看: 自身绩效 / 与v11.9的相关 / ★非red日纯增量。
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
            rows.append((pd.Timestamp(dstr),T,-(xp/ep-1)-2*FEE))
    return pd.DataFrame(rows,columns=['dt','T','r'])
P('='*140); P('★ 按入场时点拆解 (S2口径每笔0.25单位; ★13:59与v11.9的14:00决策重合)')
for code,key in KEY.items():
    s=pd.read_parquet(f'{EXT_REGIME}/output/series_{key}.parquet')
    s['date']=pd.to_datetime(s['date']); s=s.set_index('date')
    idx=s.index[s.index>=TR0]
    A=(s['strat']-s['bench']).reindex(idx).fillna(0); red=set(s[s['red']>0].index)
    d=run(code)
    P(); P(f'── {code}')
    P(f'  {"时点":<10}{"训年化":>9}{"验年化":>9}{"测年化":>9}{"":>3}'
      f'{"★纯增量训":>11}{"★纯增量验":>11}{"★纯增量测":>11}{"":>3}{"相关v119":>9}{"落red日%":>9}{"笔数":>7}')
    for T in DEC+['前三笔','全部']:
        if T=='前三笔': x=d[d['T']!=1359]
        elif T=='全部': x=d
        else: x=d[d['T']==T]
        B=(x.groupby('dt')['r'].sum()*UNIT).reindex(idx).fillna(0)
        xn=x[~x.dt.isin(red)]
        Bn=(xn.groupby('dt')['r'].sum()*UNIT).reindex(idx).fillna(0)
        vals=[]; vn=[]
        for m in [(idx>=TR0)&(idx<TRm),(idx>=TRm)&(idx<TE0),(idx>=TE0)]:
            yrs=(idx[m].max()-idx[m].min()).days/365.25
            vals.append(B[m].sum()/yrs*100); vn.append(Bn[m].sum()/yrs*100)
        xw=x[(x.dt>=TR0)]
        onred=xw.dt.isin(red).mean()*100
        lab=str(T) if not isinstance(T,str) else T
        star='★' if T==1359 else ' '
        P(f'  {lab:<8}{star:<2}{vals[0]:>+9.2f}{vals[1]:>+9.2f}{vals[2]:>+9.2f}{"":>3}'
          f'{vn[0]:>+11.2f}{vn[1]:>+11.2f}{vn[2]:>+11.2f}{"":>3}{A.corr(B):>+9.2f}{onred:>9.1f}{len(xw):>7}')
open(os.path.join(PROJ,'output','beta','by_entry_time.txt'),'w').write('\n'.join(lines))
print('saved')
