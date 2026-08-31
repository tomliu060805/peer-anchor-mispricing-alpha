"""★两种 IVU 构造的正面对比(空头腿, 5时点, k=2.0/2.5):
  IVU30 = money(09:31→10:00)/money(09:31→T) 的同时刻历史分位   —— 含义随 T 变化(占比)
  IVU*  = money(09:31→10:00) 自身的历史分位                    —— 与 T 无关(开盘是否异常活跃)
两者分位均由过去59日决定(当日仅作被比较对象), 无前视。IVU* 需完整开盘半小时 ⇒ 仅 T>=1000 可用。
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); CLOSE_HM=1457; NDAY=14; FEE=10e-4
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
DEC=[1000,1029,1129,1359]          # IVU* 需要完整开盘半小时, 故从10:00起
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)
def pct(s): return s.rolling(60,min_periods=40).apply(lambda w:(w[:-1]<w[-1]).mean(),raw=True)

def run(code,k,gate=None,thr=0.5,ver='star'):
    g=_df[_df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    pm=g.pivot_table(index='date',columns='hm',values='money')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]; cum=pm[hms].cumsum(axis=1); n30=cum.iloc[:,hp[1000]]
    star=pct(n30)                                        # 与T无关
    share={T:pct(n30/cum.iloc[:,hp[T]].replace(0,np.nan)) for T in DEC if T in hp}
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC if T in pc.columns}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]; nxt=dopen.shift(-1).values
    rows=[]
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]; ntr=0
        for T in DEC:
            if ntr>=len(DEC) or T not in sig or T not in hp: continue
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT>=o0*(1-k*s_): continue
            if gate is not None:
                q=(star.iloc[di] if ver=='star' else share[T].iloc[di])
                if not np.isfinite(q): continue
                if gate=='high' and q<=thr: continue
                if gate=='low'  and q>=thr: continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            if not np.isfinite(ep): continue
            xp=nxt[di] if np.isfinite(nxt[di]) else rc[ic]
            if not np.isfinite(xp): continue
            rows.append((dstr,-(xp/ep-1)-2*FEE)); ntr+=1
    d=pd.DataFrame(rows,columns=['date','r']); d['dt']=pd.to_datetime(d['date']); return d

def rep(d,lab):
    o=[]
    for s,m in [('训',(d.dt>=TR0)&(d.dt<TRm)),('验',(d.dt>=TRm)&(d.dt<=END))]:
        t=d[m]['r'].values
        if not len(t): o.append(f'{s} 无'); continue
        w=t>0; pfn=-t[~w].sum()
        o.append(f'{s} {len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp '
                 f'PF{(t[w].sum()/pfn if pfn>0 else np.inf):>5.2f} 合计{t.sum()*100:>+6.1f}%')
    return f'  {lab:<26}' + ' | '.join(o)

P('='*132); P('★ IVU30(占比) vs IVU*(开盘量自身分位) 正面对比 —— 空头腿, 4时点(10:00起)')
for c in ['CSI500','CSI1000','CNI2000']:
    P(); P(f'── {c}')
    for k in [2.0,2.5]:
        P(rep(run(c,k),f'k={k} 无门'))
        for ver,vn in [('star','IVU*'),('share','IVU30')]:
            for gt in ['high','low']:
                for th in [0.4,0.6]:
                    d=run(c,k,gate=gt,thr=th,ver=ver)
                    if len(d)>40: P(rep(d,f'k={k} {vn} {gt}>{th}'))
        P()
open(os.path.join(PROJ,'output','beta','ivu_star.txt'),'w').write('\n'.join(lines))
print('saved')
