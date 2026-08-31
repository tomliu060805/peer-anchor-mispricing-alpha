"""★三项决定性检验 @双边20bp (训/验决定, 测试段2024-08起封存):
  T1 多空腿分解 × k: 多头腿在更高 k 下能否活过来
  T2 空头腿收益拆解: 日内段(入场→14:57) vs 隔夜段(14:57→次日开盘)
     —— 隔夜段是结构性负漂移(任何隔夜空头都能拿), 必须从"信号价值"里扣掉
  T3 与 v11.9 的重合与增量: 触发日重合率 / red日与非red日分别的绩效 / 日收益相关
  T4 ★零信息对照: 入场时点与持有规则完全相同, 仅随机化方向(500种子, 100核)
"""
import os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); DEC=[1029,1129,1359]; CLOSE_HM=1457; NDAY=14; DPY=244
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
KEY={'CSI500':'500','CSI1000':'1000','CNI2000':'2000'}
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

def prep(code):
    g=_df[_df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns))
    return pc,po,hms,{h:i for i,h in enumerate(hms)}

def sim(code,k,maxtr=3,seed=None):
    """返回逐笔明细。seed 非空则随机化方向(零信息对照)。"""
    pc,po,hms,hp=prep(code); dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC if T in pc.columns}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    nxt=dopen.shift(-1).values; fee=10e-4
    rs=np.random.RandomState(seed) if seed is not None else None
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
            side=1 if cT>up else (-1 if cT<dn else 0)
            if side==0: continue
            if rs is not None: side=1 if rs.rand()<0.5 else -1
            j0=iT+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]; c57=rc[ic]; nо=nxt[di]
            if not np.isfinite(ep): continue
            xp = (nо if np.isfinite(nо) else c57) if side<0 else c57
            if not np.isfinite(xp): continue
            r=side*(xp/ep-1)-2*fee
            r_id = side*(c57/ep-1) if np.isfinite(c57) else np.nan          # 日内段(毛)
            r_on = side*(xp/c57-1) if (side<0 and np.isfinite(c57)) else 0.0 # 隔夜段(毛)
            rows.append((dstr,T,side,r,r_id,r_on)); ntr+=1
    d=pd.DataFrame(rows,columns=['date','T','side','r','r_id','r_on'])
    d['dt']=pd.to_datetime(d['date'])
    return d

def daily(d,idx):
    return d.groupby('dt')['r'].sum().reindex(idx).fillna(0)

def segstat(d,lab):
    o=[]
    for s,m in [('训',(d.dt>=TR0)&(d.dt<TRm)),('验',(d.dt>=TRm)&(d.dt<=END))]:
        t=d[m]['r'].values
        if not len(t): o.append(f'{s} 无'); continue
        w=t>0; pfn=-t[~w].sum()
        o.append(f'{s} {len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp '
                 f'PF{(t[w].sum()/pfn if pfn>0 else np.inf):>4.2f} 合计{t.sum()*100:>+6.1f}%')
    return f'  {lab:<28}' + ' | '.join(o)

CODES=['CSI500','CSI1000','CNI2000']
P('='*146); P('★ 决定性检验 @双边20bp (所有数字均为【扣费后净值】; 训/验决定, 测试段封存)')

P(); P('【T1 多空腿分解 × k —— 多头腿在高 k 下能否活过来】')
for c in CODES:
    P(f'  ── {c}')
    for k in [1.0,2.0,2.5,3.0]:
        d=sim(c,k)
        for sd,nm in [(1,'多头'),(-1,'空头')]:
            P(segstat(d[d.side==sd],f'k={k} {nm}腿'))

P(); P('【T2 空头腿收益拆解: 日内段 vs 隔夜段(均为毛收益, 未扣费)】')
P('  ★隔夜段是A股结构性负漂移, 任何隔夜空头都能拿到 —— 不属于信号的功劳')
for c in CODES:
    for k in [2.0,2.5]:
        d=sim(c,k); s=d[d.side==-1]
        if not len(s): continue
        P(f'  {c:<9} k={k}  空头 {len(s):>4}笔 | 日内段 {s.r_id.mean()*1e4:>+6.1f}bp | '
          f'隔夜段 {s.r_on.mean()*1e4:>+6.1f}bp | 毛合计 {(s.r_id+s.r_on).mean()*1e4:>+6.1f}bp '
          f'⇒ 隔夜占比 {s.r_on.mean()/max((s.r_id+s.r_on).mean(),1e-12)*100:>4.0f}%')

P(); P('【T3 与 v11.9 的重合与增量】')
for c in CODES:
    sr=pd.read_parquet(f'{EXT_REGIME}/output/series_{KEY[c]}.parquet')
    sr['date']=pd.to_datetime(sr['date']); sr=sr.set_index('date')
    red=set(sr[sr['red']>0].index)
    for k in [2.0,2.5]:
        d=sim(c,k)
        d['isred']=d.dt.isin(red)
        W=(d.dt>=TR0)&(d.dt<=END)
        dd=d[W]
        ov=dd['isred'].mean()*100
        P(f'  {c:<9} k={k}  触发笔中落在red日 {ov:>4.1f}%')
        for tag,mm in [('red日',dd.isred),('非red日',~dd.isred)]:
            t=dd[mm]['r'].values
            if len(t):
                w=t>0; pfn=-t[~w].sum()
                P(f'      {tag:<7} {len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp '
                  f'PF{(t[w].sum()/pfn if pfn>0 else np.inf):>4.2f}')
        idx=sr.index[(sr.index>=TR0)&(sr.index<=END)]
        mine=daily(d,idx); v119=(sr['strat']-sr['bench']).reindex(idx).fillna(0)
        P(f'      日收益与v11.9超额相关 {mine.corr(v119):+.3f}')

def one_seed(a):
    c,k,s=a
    d=sim(c,k,seed=s)
    W=(d.dt>=TR0)&(d.dt<=END)
    return d[W]['r'].sum()*100

P(); P('【T4 ★零信息对照: 入场时点/持有规则完全相同, 仅随机化方向(500种子)】')
for c in CODES:
    for k in [2.0,2.5]:
        d=sim(c,k); W=(d.dt>=TR0)&(d.dt<=END)
        real=d[W]['r'].sum()*100
        with ProcessPoolExecutor(max_workers=100) as ex:
            sims=np.array(list(ex.map(one_seed,[(c,k,s) for s in range(500)])))
        z=(real-sims.mean())/sims.std()
        P(f'  {c:<9} k={k}  真实合计{real:>+7.1f}%  随机方向: 均值{sims.mean():>+7.1f}% 标准差{sims.std():>5.1f}% '
          f'95分位{np.percentile(sims,95):>+7.1f}%  ⇒ z={z:>5.2f}, 超越{(sims<real).mean()*100:>5.1f}%')
open(os.path.join(PROJ,'output','beta','decisive.txt'),'w').write('\n'.join(lines))
print('\nsaved output/decisive.txt')
