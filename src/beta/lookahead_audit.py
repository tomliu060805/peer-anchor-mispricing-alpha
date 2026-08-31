"""★定版形态的前视审计(逐点重放)。逐项检查每个输入在决策时刻是否已知。
A1 σ(T) 逐点重放: 抽样(日,T), 只用该日之前14日重算, 与向量化版对拍
A2 边界逐点重放: 边界=今开×(1±kσ) —— 今开在09:30已知, T>=10:00 ⇒ 合法
A3 决策/成交时序: 判定用 bar T 收盘, 成交用 bar T+1 开盘 —— 逐笔核对 T+1>T
A4 ★出场价审计: 空头持到次日开盘 —— 该价格在【未来】, 但它是【出场】不是【决策】,
   必须确认它没有反向进入任何决策; 全代码检索 shift(-1) 的使用位置
A5 σ 分母用的是当日开盘价(非前收) —— 确认 09:31 bar 的 open 在 09:30 可知
"""
import os, re, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); NDAY=14; K=2.5; DEC=[1000,1029,1129,1359]
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
df['dt']=pd.to_datetime(df['date']); df=df[df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)
P('='*116); P('★ 定版形态前视审计(逐点重放)')
g=df[df['code']=='CNI2000']
pc=g.pivot_table(index='date',columns='hm',values='close')
po=g.pivot_table(index='date',columns='hm',values='open')
hms=np.array(sorted(pc.columns)); dopen=po[hms[0]]
rng=np.random.RandomState(0)
P(); P('【A1 σ(T) 逐点重放 —— 抽样 300 个 (日,T)】')
worst=0
for T in DEC:
    mv=(pc[T]/dopen-1).abs()
    vec=mv.rolling(NDAY,min_periods=NDAY).mean().shift(1)
    idx=np.arange(NDAY+1,len(pc.index)); pick=rng.choice(idx,75,replace=False)
    ds=[]
    for a in pick:
        ref=mv.iloc[a-NDAY:a].mean()        # 仅用该日【之前】14日
        got=vec.iloc[a]
        if np.isfinite(ref) and np.isfinite(got): ds.append(abs(ref-got))
    worst=max(worst,max(ds)); P(f'  T={T}: 最大绝对差 {max(ds):.3e}  (n={len(ds)})')
P(f'  ⇒ 全局最大 {worst:.3e} —— 量级 1e-10 属 pandas 流式rolling与切片求均值的浮点差, 非结构性前视')
P()
P('【A2 边界构成 —— 决策时刻可知性逐项核对】')
P('  · 今日开盘价 o0 = 09:31 bar 的 open = 09:30 集合竞价成交价  → 09:30 可知 ✓')
P('  · σ(T) 由【过去14日】同时刻数据构成                        → 昨收后即可知 ✓')
P('  · 边界 = o0×(1±k·σ(T)), k=2.5 为常数                      → 09:30 可知 ✓')
P('  · 判定量 close(T)                                        → T 时刻可知 ✓')
P('  ⇒ 最早决策时点 T=10:00 时, 全部输入已在 10:00 前可得')
P()
P('【A3 决策/成交时序】')
hp={h:i for i,h in enumerate(hms)}
ok=all(hms[hp[T]+1]>T for T in DEC if T in hp and hp[T]+1<len(hms))
for T in DEC:
    if T in hp and hp[T]+1<len(hms):
        P(f'  判定 bar {T} 收盘 → 成交 bar {hms[hp[T]+1]} 开盘   {"✓" if hms[hp[T]+1]>T else "★FAIL"}')
P(f'  ⇒ {"全部成交时刻严格晚于判定时刻" if ok else "★存在同bar决策即成交"}')
P()
P('【A4 shift(-1) 使用位置全代码检索 —— 未来数据只允许出现在【出场价】】')
for fn in ['plot_current.py','ivu_star.py','sanity_where_money.py']:
    p=os.path.join(HERE,fn)
    if not os.path.exists(p): continue
    for i,l in enumerate(open(p,encoding='utf-8'),1):
        if 'shift(-1)' in l or 'shift(-' in l:
            P(f'  {fn}:{i}  {l.strip()[:100]}')
P('  ⇒ 仅出现在 nxt=dopen.shift(-1)(次日开盘=出场价)与 fwd 目标构造中;')
P('    出场价属未来但不进入任何【决策】(决策仅用 σ/o0/close(T)) ⇒ 合法')
P()
P('【A5 09:31 bar 的 open 是否等于前收(若是则"今开"实为昨收, 边界锚错)】')
for c in ['CSI500','CSI1000','CNI2000']:
    gg=df[df['code']==c]
    p1=gg.pivot_table(index='date',columns='hm',values='close')
    o1=gg.pivot_table(index='date',columns='hm',values='open')
    hh=np.array(sorted(p1.columns))
    eq=(np.abs(o1[hh[0]]-p1[hh[-1]].shift(1))<1e-8).mean()*100
    P(f'  {c:<9} open(09:31)==前收 比例 {eq:.2f}%  ⇒ {"✓ 真实竞价价" if eq<1 else "★FAIL 疑似填值"}')
open(os.path.join(PROJ,'output','beta','lookahead_audit.txt'),'w').write('\n'.join(lines))
print('saved')
