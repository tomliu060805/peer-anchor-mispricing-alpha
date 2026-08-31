"""★数据完整性 + 收益来源解剖 —— 回答"这钱到底从哪来, 是不是假的"。
D1 ★致命性检查: 指数 09:31 的 open 是否等于前收盘(pre_close填值 ⇒ 隔夜跳空是假的)
D2 收益按时段解剖: 入场→14:57(日内) / 14:57→次日09:31open(隔夜) 各贡献多少
D3 隔夜段与"指数陈旧"的关系: 次日 09:31open→09:35 的走势(若开盘价陈旧, 会在开盘后几分钟补回)
    —— 若空头在开盘平仓后, 价格立刻反向补回, 说明吃的是陈旧价差, 不可交易
D4 持仓时长分布与多空腿完整画像
"""
import os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); DEC=[1029,1129,1359]; CLOSE_HM=1457; NDAY=14; FEE=10e-4
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

P('='*130); P('★ D1 致命性检查: 指数 09:31 的 open 是否 = 前一日收盘(若是则隔夜跳空为假)')
for c in ['CSI500','CSI1000','CNI2000']:
    g=_df[_df['code']==c]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns))
    o931=po[hms[0]]; last=pc[hms[-1]]
    prev=last.shift(1)
    gap=(o931/prev-1)
    eq=(np.abs(o931-prev)<1e-8).mean()*100
    P(f'  {c:<9} open(首bar)==前收 的比例 {eq:>5.2f}%  | 跳空均值{gap.mean()*1e4:>+6.2f}bp '
      f'标准差{gap.std()*1e4:>6.1f}bp  |首bar收盘时刻 {hms[0]} 末bar {hms[-1]}')
P('  ⇒ 若比例接近0%, 开盘价是真实竞价价格, 隔夜跳空可信')

P(); P('='*130); P('★ D2/D3 收益解剖 (只空头腿, k=2.0, 3时点)')
for c in ['CSI500','CSI1000','CNI2000']:
    g=_df[_df['code']==c]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    nopen=dopen.shift(-1).values
    # 次日开盘后 4 分钟(09:35) 收盘, 用于检验陈旧价补回
    i935=hp.get(935)
    n935=pc[935].shift(-1).values if 935 in pc.columns else None
    rows=[]
    for di in range(len(pc.index)):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]; ntr=0
        for T in DEC:
            if ntr>=3 or T not in hp: continue
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT>=o0*(1-2.0*s_): continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]; c57=rc[ic]; no=nopen[di]; n5=n935[di] if n935 is not None else np.nan
            if not (np.isfinite(ep) and np.isfinite(c57) and np.isfinite(no)): continue
            rows.append(dict(dt=pd.Timestamp(pc.index[di]),T=T,
                hold_min=(hms[ic]//100*60+hms[ic]%100)-(hms[j0]//100*60+hms[j0]%100),
                r_id=-(c57/ep-1), r_on=-(no/c57-1),
                r_next5=-(n5/no-1) if np.isfinite(n5) else np.nan,
                r_net=-(no/ep-1)-2*FEE))
            ntr+=1
    d=pd.DataFrame(rows)
    for lab,m in [('训',(d.dt>=TR0)&(d.dt<TRm)),('验',(d.dt>=TRm)&(d.dt<=END))]:
        x=d[m]
        P(f'  {c:<9}{lab} {len(x):>4}笔 | 日内段{x.r_id.mean()*1e4:>+6.1f}bp '
          f'隔夜段{x.r_on.mean()*1e4:>+6.1f}bp 净{x.r_net.mean()*1e4:>+6.1f}bp | '
          f'★次日开盘后4分钟(继续持空){x.r_next5.mean()*1e4:>+6.1f}bp '
          f'(若显著为负 ⇒ 开盘价陈旧, 平仓后立刻反向补回)')

P(); P('='*130); P('★ D4 持仓时长与多空腿画像 (k=2.0, 3时点)')
for c in ['CSI500','CSI1000','CNI2000']:
    g=_df[_df['code']==c]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    nopen=dopen.shift(-1).values; rows=[]
    for di in range(len(pc.index)):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]; ntr=0
        for T in DEC:
            if ntr>=3 or T not in hp: continue
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            side=1 if cT>o0*(1+2.0*s_) else (-1 if cT<o0*(1-2.0*s_) else 0)
            if side==0: continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            xp=(nopen[di] if np.isfinite(nopen[di]) else rc[ic]) if side<0 else rc[ic]
            if not (np.isfinite(ep) and np.isfinite(xp)): continue
            hm_e=hms[j0]; mins=(hms[ic]//100*60+hms[ic]%100)-(hm_e//100*60+hm_e%100)
            if hm_e>1130: mins-=90                      # 扣午休
            rows.append(dict(dt=pd.Timestamp(pc.index[di]),side=side,T=T,
                hold_min=mins + (17*60+31-15*0 if side<0 else 0),   # 空头再加隔夜(约收盘→次日开盘)
                r=side*(xp/ep-1)-2*FEE)); ntr+=1
    d=pd.DataFrame(rows)
    P(f'  ── {c}')
    for sd,nm in [(-1,'空头'),(1,'多头')]:
        x=d[d.side==sd]
        for lab,m in [('训',(x.dt>=TR0)&(x.dt<TRm)),('验',(x.dt>=TRm)&(x.dt<=END))]:
            y=x[m]
            if not len(y): continue
            w=y.r>0; pfn=-y.r[~w].sum()
            hold_id=y[y['T']<=1359]['hold_min']
            dist='/'.join(f"{t}:{int((y['T']==t).sum())}" for t in DEC)
            hm_med=hold_id.median()-(17*60+31 if sd<0 else 0)
            P(f'    {nm} {lab} {len(y):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{y.r.mean()*1e4:>+6.1f}bp '
              f'PF{(y.r[w].sum()/pfn if pfn>0 else np.inf):>5.2f} | 日内持仓中位{hm_med:>4.0f}分钟 | 入场分布 '+dist)
open(os.path.join(PROJ,'output','beta','sanity_where_money.txt'),'w').write('\n'.join(lines))
print('\nsaved output/sanity_where_money.txt')
