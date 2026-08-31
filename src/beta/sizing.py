"""★仓位口径对比 (定版形态: CNI2000/CSI1000/CSI500, k=2.5, 4时点, 只空头, 持隔夜, 20bp)。
同日最多4笔且互相重叠, 不同资金分配方式的绩效与敞口:
  S1 等分资金      每笔 1/n (n=当日笔数)        → 最大敞口 1x, 有信号日恒满仓
  S2 固定1/4       每笔 0.25 单位               → 最大敞口 1x, 单信号日仅25%仓
  S3 固定1/2       每笔 0.5  单位               → 最大敞口 2x
  S4 每信号1单位    每笔 1 单位                  → 最大敞口 4x
  S5 每信号1单位封顶2x                          → 折中
★Sharpe 对线性缩放不变, 故 S2/S3/S4/S5 的 Sharpe 相同; 与 S1 的差异才是【信号数是否含信息】。
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); CLOSE_HM=1457; NDAY=14; FEE=10e-4; K=2.5
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15'); DEC=[1000,1029,1129,1359]
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
df['dt']=pd.to_datetime(df['date']); df=df[df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

def trades(code):
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
    d=pd.DataFrame(rows,columns=['dt','r'])
    return d, pd.to_datetime(pc.index)

def stat(r,expo,lab):
    W=(r.index>=TR0)&(r.index<=END); r=r[W]; e=expo[W]
    ann=r.mean()*244*100; vol=r.std()*np.sqrt(244)*100
    eq=(1+r).cumprod(); mdd=((eq/eq.cummax())-1).min()*100
    tm=(r.index>=TR0)&(r.index<TRm); vm=(r.index>=TRm)&(r.index<=END)
    return (f'  {lab:<20} 年化{ann:>+6.2f}% SR{(ann/vol if vol>0 else np.nan):>5.2f} MDD{mdd:>7.2f}% '
            f'均敞口{e.mean():>5.2f} 峰值{e.max():>4.1f} | 训{r[tm].mean()*244*100:>+6.2f}% 验{r[vm].mean()*244*100:>+6.2f}% '
            f'| 收益/均敞口{ann/max(e.mean(),1e-9):>6.2f}')

for code in ['CSI500','CSI1000','CNI2000']:
    d,idx=trades(code)
    g=d.groupby('dt')['r'].agg(['sum','mean','size'])
    n=g['size'].reindex(idx).fillna(0)
    P(); P(f'── {code}  (有仓日 {int((n>0).sum())}, 同日笔数分布 {g["size"].value_counts().sort_index().to_dict()})')
    for lab,ret,ex in [
        ('S1 等分资金(1/n)',   g['mean'].reindex(idx).fillna(0), (n>0).astype(float)),
        ('S2 固定1/4每笔',     g['sum'].reindex(idx).fillna(0)*0.25, n*0.25),
        ('S3 固定1/2每笔',     g['sum'].reindex(idx).fillna(0)*0.5,  n*0.5),
        ('S4 每信号1单位',      g['sum'].reindex(idx).fillna(0),      n*1.0),
        ('S5 每信号1单位封顶2x', (g['sum']/g['size']*np.minimum(g['size'],2)).reindex(idx).fillna(0), np.minimum(n,2.0)),
    ]:
        P(stat(ret,ex,lab))
P(); P('★ 读法: Sharpe 对线性缩放不变 ⇒ S2/S3/S4 同 SR; S1 与它们的差异才反映"同日信号数是否含信息"。')
P('  末列"收益/均敞口"= 单位敞口的年化产出, 是跨口径可比的效率指标。')
open(os.path.join(PROJ,'output','beta','sizing.txt'),'w').write('\n'.join(lines))
print('saved')
