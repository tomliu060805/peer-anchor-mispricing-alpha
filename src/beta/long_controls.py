"""★多日持有多头腿的决定性对照 —— 排除"只是beta"。
B1 敞口匹配的被动基准: 以相同平均敞口被动持有指数(买入持有×敞口)
B2 ★随机入场对照: 相同笔数、相同持有期, 入场日随机(500种子) —— 若策略不显著优于它, 信号无价值
B3 超额分解: 策略日收益 − 敞口×指数日收益 = 择时超额
口径: 每笔0.25单位, 持有N日(重叠), 双边20bp; 训/验决定, 测试段已开封仅列观察。
"""
import os, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
CLOSE_HM=1457; NDAY=14; FEE=10e-4; UNIT=0.25; DEC=[1000,1029,1129,1359]; K=2.0
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
    ic=np.where(hms<=CLOSE_HM)[0][-1]
    idx=pd.to_datetime(pc.index)
    return pc,po,hms,hp,po[hms[0]],pc.iloc[:,ic].values,idx

def entries(code,k):
    pc,po,hms,hp,dopen,c57,idx=prep(code)
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC}
    C,O=pc.values,po.values; out=[]
    for di in range(len(idx)):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]
        for T in DEC:
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT<=o0*(1+k*s_): continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            if np.isfinite(ro[j0]): out.append((di,ro[j0]))
    return out,c57,idx

def series(ent,c57,idx,N):
    """逐日收益(重叠仓位, 每笔UNIT)与逐日敞口。收益按持有期均摊到每一天。"""
    r=np.zeros(len(idx)); e=np.zeros(len(idx))
    for di,ep in ent:
        xi=di+N
        if xi>=len(idx) or not np.isfinite(c57[xi]): continue
        tot=(c57[xi]/ep-1)-2*FEE
        span=max(xi-di,1)
        r[di+1:xi+1]+=tot/span*UNIT
        e[di+1:xi+1]+=UNIT
    return pd.Series(r,index=idx),pd.Series(e,index=idx)

def seg(x,idx):
    o={}
    for nm,a,b in [('训',TR0,TRm),('验',TRm,TE0),('测*',TE0,pd.Timestamp('2100-01-01'))]:
        m=(idx>=a)&(idx<b)
        if m.sum()<50: continue
        yrs=(idx[m].max()-idx[m].min()).days/365.25
        o[nm]=x[m].sum()/yrs*100
    return o

def rnd_one(a):
    code,k,N,seed=a
    ent,c57,idx=ENT[code]
    rs=np.random.RandomState(seed)
    di=rs.choice(len(idx)-N-1,size=len(ent),replace=True)
    pc,po,hms,hp,dopen,_,_=prep(code)
    ro=po.values
    fake=[]
    for d in di:
        j=hp[DEC[rs.randint(len(DEC))]]+1
        if j<len(hms) and np.isfinite(ro[d][j]): fake.append((d,ro[d][j]))
    r,_=series(fake,c57,idx,N)
    s=seg(r,idx)
    return s.get('训',np.nan),s.get('验',np.nan)

ENT={}
P('='*140); P('★ 多日持有多头腿 决定性对照 (k=2.0, 每笔0.25单位, 双边20bp)')
for code in ['CSI1000','CNI2000']:
    ent,c57,idx=entries(code,K); ENT[code]=(ent,c57,idx)
    bench=pd.Series(np.r_[np.nan,c57[1:]/c57[:-1]-1],index=idx).fillna(0)
    P(); P(f'── {code}  入场 {len(ent)} 笔')
    P(f'  {"持有":<8}{"策略训":>9}{"策略验":>9}{"均敞口":>8}{"敞口匹配被动 训":>16}{"敞口匹配被动 验":>16}{"★择时超额 训":>14}{"★择时超额 验":>14}')
    for N in [1,3,5,10]:
        r,e=series(ent,c57,idx,N)
        pas=bench*e                                     # 敞口匹配的被动持有
        ex=r-pas
        sr,sp,sx=seg(r,idx),seg(pas,idx),seg(ex,idx)
        P(f'  {N:<8}{sr.get("训",np.nan):>+9.2f}{sr.get("验",np.nan):>+9.2f}{e.mean():>8.3f}'
          f'{sp.get("训",np.nan):>+16.2f}{sp.get("验",np.nan):>+16.2f}{sx.get("训",np.nan):>+14.2f}{sx.get("验",np.nan):>+14.2f}')

P(); P('★ B2 随机入场对照 (相同笔数与持有期, 入场日随机, 200种子)')
for code in ['CSI1000','CNI2000']:
    ent,c57,idx=ENT[code]
    for N in [1,10]:
        r,_=series(ent,c57,idx,N); s=seg(r,idx)
        with ProcessPoolExecutor(max_workers=60) as ex_:
            sims=np.array(list(ex_.map(rnd_one,[(code,K,N,sd) for sd in range(200)])))
        for j,nm in [(0,'训'),(1,'验')]:
            v=sims[:,j]; real=s.get(nm,np.nan)
            z=(real-np.nanmean(v))/np.nanstd(v)
            P(f'  {code:<9} N={N:<3} {nm}: 真实{real:>+7.2f}%  随机均值{np.nanmean(v):>+7.2f}% 标准差{np.nanstd(v):>5.2f}% '
              f'⇒ z={z:>5.2f}, 超越{(v<real).mean()*100:>5.1f}%')
open(os.path.join(PROJ,'output','beta','long_controls.txt'),'w').write('\n'.join(lines))
print('saved')
