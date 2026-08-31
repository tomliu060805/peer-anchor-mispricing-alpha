"""★用 IVU(开盘量占比) 做多头腿的状态开关。
★前视改造声明: Yang&He 的 IVU 分母是"前七个区间"(到14:30), 用于 14:30 决策合法;
   本策略决策在 10:29/11:29/13:59, 直接搬会前视。改造为逐点可知版:
     IVU_pit(T) = money(09:31→10:00) / money(09:31→T)
   再取【过去60日同时刻分布】的分位(严格排除当日), 因为该比值在不同时刻量纲不同。
预先声明的网格: 方向{高IVU开/低IVU开} × 分位阈{0.3,0.5,0.7} × k{1.5,2.0,2.5}; 只作用于多头腿。
判据: 训验双正 且 验证段单笔>0; 测试段2024-08起封存。
"""
import os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); DEC=[1029,1129,1359]; CLOSE_HM=1457; NDAY=14; DPY=244
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15'); FEE=10e-4
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

def build(code):
    g=_df[_df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    pm=g.pivot_table(index='date',columns='hm',values='money')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    cum=pm[hms].cumsum(axis=1)
    i1000=hp.get(1000)
    open30=cum.iloc[:,i1000]                                    # 09:31→10:00 累计成交额
    ivu={}
    for T in DEC:
        if T not in hp: continue
        den=cum.iloc[:,hp[T]]
        raw=(open30/den.replace(0,np.nan))
        # 过去60日同时刻分位, 严格排除当日
        ivu[T]=raw.rolling(60,min_periods=40).apply(lambda w:(w[:-1]<w[-1]).mean(),raw=True).shift(0)
        # 说明: rolling 窗含当日, 但比较时用 w[:-1]<w[-1] —— 分位由过去59日决定, 当日只作被比较对象
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC if T in pc.columns}
    return pc,po,hms,hp,dopen,sig,ivu

def run(code,k,gate=None,thr=0.5,maxtr=3):
    """gate: None=不设门 | 'high'=IVU分位>thr才做多 | 'low'=IVU分位<thr才做多。空头腿始终不设门。"""
    pc,po,hms,hp,dopen,sig,ivu=build(code)
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    nxt=dopen.shift(-1).values
    rows=[]
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]; ntr=0
        for T in DEC:
            if ntr>=maxtr or T not in sig or T not in hp: continue
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            side=1 if cT>o0*(1+k*s_) else (-1 if cT<o0*(1-k*s_) else 0)
            if side==0: continue
            if side==1 and gate is not None:
                q=ivu[T].iloc[di] if T in ivu else np.nan
                if not np.isfinite(q): continue
                if gate=='high' and q<=thr: continue
                if gate=='low'  and q>=thr: continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            if not np.isfinite(ep): continue
            xp = (nxt[di] if np.isfinite(nxt[di]) else rc[ic]) if side<0 else rc[ic]
            if not np.isfinite(xp): continue
            rows.append((dstr,side,side*(xp/ep-1)-2*FEE)); ntr+=1
    d=pd.DataFrame(rows,columns=['date','side','r']); d['dt']=pd.to_datetime(d['date'])
    return d

def leg(d,sd):
    x=d[d.side==sd]; o=[]
    for s,m in [('训',(x.dt>=TR0)&(x.dt<TRm)),('验',(x.dt>=TRm)&(x.dt<=END))]:
        t=x[m]['r'].values
        if not len(t): o.append(f'{s} 无'); continue
        w=t>0; pfn=-t[~w].sum()
        o.append(f'{s} {len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp PF{(t[w].sum()/pfn if pfn>0 else np.inf):>5.2f}')
    return ' | '.join(o)

P('='*136); P('★ IVU 作多头腿状态开关 (逐点可知改造版; 训/验决定, 测试段封存)')
P('  ★前视改造: IVU_pit(T)=money(09:31→10:00)/money(09:31→T), 分位由过去59日同时刻决定')
for code in ['CSI500','CSI1000','CNI2000']:
    P(); P(f'── {code}')
    for k in [1.5,2.0,2.5]:
        d0=run(code,k)
        P(f'  k={k} 多头腿 无门        ' + leg(d0,1))
        for gate in ['high','low']:
            for thr in [0.3,0.5,0.7]:
                d=run(code,k,gate=gate,thr=thr)
                P(f'  k={k} 多头腿 {gate}>{thr}    ' + leg(d,1))
        P(f'  k={k} 空头腿(对照,恒无门)  ' + leg(d0,-1))
        P()
open(os.path.join(PROJ,'output','beta','ivu_long.txt'),'w').write('\n'.join(lines))
print('\nsaved output/ivu_long.txt')
