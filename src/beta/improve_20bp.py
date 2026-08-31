"""★20bp 下的改进检验。预先声明的网格与假设(先写后跑, 不回头改):
  H1 入场时点: 单笔毛边际随入场时刻单调递减(晚入场剩余时间少)
  H2 突破幅度 k∈{1.0,1.25,1.5,2.0,2.5}: 更大突破 ⇒ 更高单笔边际
  H3 止损: 去掉边界止损(持到收盘)应提高单笔毛边际(止损在高成本下是负资产)
  H4 ★空头持隔夜: A股隔夜漂移为负(对空头是顺风), 多头14:57平/空头持到次日开盘
判据: 双边20bp下 训(2016-01~2022-02-14) 与 验(2022-02-15~2024-07-31) 双正; 测试段2024-08起封存。
铁律指标: 触发数/胜率/单笔均/盈亏比/盈利因子。
"""
import os, sys, itertools
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); DEC=[1029,1129,1359]; CLOSE_HM=1457; NDAY=14; DPY=244
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
df['dt']=pd.to_datetime(df['date']); df=df[df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

def panels(g):
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    return pc,po,np.array(sorted(pc.columns))

def run(g, fee_bp, k=1.0, use_stop=True, max_trades=3, on_short=False, dec_times=None):
    fee=fee_bp*1e-4; dts=dec_times or DEC
    pc,po,hms=panels(g); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1)
         for T in dts if T in pc.columns}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    nxt_open=dopen.shift(-1).values                      # 次日开盘(空头隔夜用)
    rets=[]; recs=[]
    for di in range(len(pc.index)):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): rets.append(0.0); continue
        rc,ro=C[di],O[di]; day=0.0; ntr=0
        for T in dts:
            if ntr>=max_trades or T not in sig or T not in hp: continue
            s=sig[T].iloc[di]
            if not np.isfinite(s): continue
            iT=hp[T]; cT=rc[iT]
            if not np.isfinite(cT): continue
            up,dn=o0*(1+k*s), o0*(1-k*s)
            side=1 if cT>up else (-1 if cT<dn else 0)
            if side==0: continue
            j0=iT+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            if not np.isfinite(ep): continue
            xp=None
            if use_stop:
                for j in range(j0,len(hms)):
                    if hms[j]>CLOSE_HM: break
                    cj=rc[j]
                    if not np.isfinite(cj): continue
                    st=dn if side>0 else up
                    if ((cj<st) if side>0 else (cj>st)) and j+1<len(hms) and hms[j+1]<=CLOSE_HM:
                        xp=ro[j+1]; break
            if xp is None:
                if side<0 and on_short:
                    xp=nxt_open[di]                       # 空头持到次日开盘
                    if not np.isfinite(xp): xp=rc[ic]
                else:
                    xp=rc[ic]
            if np.isfinite(xp):
                r=side*(xp/ep-1)-2*fee
                day+=r; ntr+=1
                recs.append((pc.index[di],T,side,r))
        rets.append(day)
    return pd.Series(rets,index=pd.to_datetime(pc.index)), pd.DataFrame(recs,columns=['date','T','side','r'])

def tm(t):
    if not len(t): return '无交易'
    w=t>0; aw=t[w].mean() if w.any() else 0; al=t[~w].mean() if (~w).any() else 0
    pf=t[w].sum()/abs(t[~w].sum()) if (w.any() and (~w).any()) else np.inf
    return (f'{len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp '
            f'盈亏比{abs(aw/al) if al else np.inf:>5.2f} PF{pf:>5.2f}')

def seg(r, tr_df):
    o=[]
    for lab,m in [('训',(r.index>=TR0)&(r.index<TRm)),('验',(r.index>=TRm)&(r.index<=END))]:
        rr=r[m]; ann=rr.mean()*DPY*100; vol=rr.std()*np.sqrt(DPY)*100
        d=pd.to_datetime(tr_df['date']) if len(tr_df) else pd.Series([],dtype='datetime64[ns]')
        t=tr_df['r'].values[((d>=TR0)&(d<TRm)).values] if lab=='训' else tr_df['r'].values[((d>=TRm)&(d<=END)).values]
        o.append(f'{lab} 年化{ann:>+7.2f}% SR{(ann/vol if vol>0 else np.nan):>5.2f} {tm(t)}')
    return ' | '.join(o)

CODES=['CSI500','CSI1000','CNI2000']
FEE=10.0   # 单边10bp = 双边20bp
P('='*150); P('★ 20bp 下的改进检验 (训 2016-01~2022-02-14 / 验 2022-02-15~2024-07-31; 测试段封存)')

P(); P('【H1 入场时点分解 —— 单笔毛边际是否随入场时刻递减】(基线规则, 毛边际=净+20bp)')
for c in CODES:
    r,d=run(df[df['code']==c],FEE)
    row=[]
    for T in DEC:
        t=d[d['T']==T]['r'].values
        row.append(f'{T}: {len(t):>4}笔 毛{(t.mean()+2e-3)*1e4:>+6.1f}bp' if len(t) else f'{T}: -')
    P(f'  {c:<9}' + '   '.join(row))

P(); P('【H2 突破幅度 k 扫描】(每日最多3笔, 有止损)')
for c in CODES:
    P(f'  ── {c}')
    for k in [1.0,1.25,1.5,2.0,2.5]:
        r,d=run(df[df['code']==c],FEE,k=k)
        gross=(d['r'].mean()+2e-3)*1e4 if len(d) else np.nan
        P(f'    k={k:<5} 毛单笔{gross:>+6.1f}bp | ' + seg(r,d))

P(); P('【H3 去掉止损(持到收盘)】k=1.0')
for c in CODES:
    r,d=run(df[df['code']==c],FEE,use_stop=False)
    gross=(d['r'].mean()+2e-3)*1e4 if len(d) else np.nan
    P(f'  {c:<9} 无止损 毛单笔{gross:>+6.1f}bp | ' + seg(r,d))

P(); P('【H4 ★空头持隔夜(多头14:57平/空头次日开盘平)】k=1.0, 无止损')
for c in CODES:
    r,d=run(df[df['code']==c],FEE,use_stop=False,on_short=True)
    gross=(d['r'].mean()+2e-3)*1e4 if len(d) else np.nan
    P(f'  {c:<9} 毛单笔{gross:>+6.1f}bp | ' + seg(r,d))
    for sd,nm in [(1,'多头'),(-1,'空头')]:
        t=d[d['side']==sd]['r'].values
        P(f'      {nm}: {tm(t)}  毛单笔{(t.mean()+2e-3)*1e4 if len(t) else np.nan:>+6.1f}bp')

P(); P('【H2×H3×H4 组合: 只做首个决策点 + 大突破 + 无止损 + 空头隔夜】')
for c in CODES:
    P(f'  ── {c}')
    for k in [1.5,2.0,2.5]:
        r,d=run(df[df['code']==c],FEE,k=k,use_stop=False,on_short=True,max_trades=1)
        gross=(d['r'].mean()+2e-3)*1e4 if len(d) else np.nan
        P(f'    k={k:<5} 毛单笔{gross:>+6.1f}bp | ' + seg(r,d))
open(os.path.join(PROJ,'output','beta','improve_20bp.txt'),'w').write('\n'.join(lines))
print('\nsaved output/improve_20bp.txt')
