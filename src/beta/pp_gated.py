"""★用户提法: pp≥0.90(早盘尾部已被打穿, 10:00即可知) 作为前提, 再用噪声突破做【早进场】扳机。
与 v11.9 的区别: v11.9 需再过 gateC(09:31→14:00<0) 且等到14:00才动手, 让出了 10:00→14:00 这段。
时序合法性: pp=spct(−mkt_q10,252,120), mkt_q10 由 09:31→10:00 个股收益构成 ⇒ 10:00 可知。
对照组:
  C1 pp≥0.90 ∧ 突破     (用户提法)
  C2 pp≥0.90 无突破要求  (每个时点都做空 —— 检验突破是否必要)
  C3 突破 无pp要求        (原版 —— 检验pp是否必要)
★增量口径: 在 red=0 的日子上的表现(v11.9 在那些日子空仓) —— 这才是真正的新东西。
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
CLOSE_HM=1457; NDAY=14; FEE=10e-4; K=2.5; UNIT=0.25
EARLY=[1000,1029,1129]                     # 早盘三笔(不含与v11.9重合的13:59)
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15'); TE0=pd.Timestamp('2024-08-01')
KEY={'CSI500':'500','CSI1000':'1000','CNI2000':'2000'}
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet')); df['dt']=pd.to_datetime(df['date'])
lines=[]
def P(s=''):
    print(s); lines.append(s)

def sim(code,mode,ppmin=0.90):
    s=pd.read_parquet(f'{EXT_REGIME}/output/series_{KEY[code]}.parquet')
    s['date']=pd.to_datetime(s['date']); s=s.set_index('date')
    pp=s['pp']; red=s['red']
    g=df[df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in EARLY}
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]; nxt=dopen.shift(-1).values
    rows=[]
    for di,dstr in enumerate(pc.index):
        d0=pd.Timestamp(dstr); o0=dopen.iloc[di]
        if not np.isfinite(o0) or d0 not in pp.index: continue
        p=pp.loc[d0]; rd=red.loc[d0]
        rc,ro=C[di],O[di]
        for T in EARLY:
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            brk = cT < o0*(1-K*s_)
            if mode=='C1' and not (np.isfinite(p) and p>=ppmin and brk): continue
            if mode=='C2' and not (np.isfinite(p) and p>=ppmin): continue
            if mode=='C3' and not brk: continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]; xp=nxt[di] if np.isfinite(nxt[di]) else rc[ic]
            if not (np.isfinite(ep) and np.isfinite(xp)): continue
            rows.append((d0,T,-(xp/ep-1)-2*FEE, rd>0))
    return pd.DataFrame(rows,columns=['dt','T','r','isred']), s

def rep(d,idx,lab,onlynonred=False):
    x=d[~d.isred] if onlynonred else d
    B=(x.groupby('dt')['r'].sum()*UNIT).reindex(idx).fillna(0)
    o=[]
    for nm,m in [('训',(idx>=TR0)&(idx<TRm)),('验',(idx>=TRm)&(idx<TE0)),('测',(idx>=TE0))]:
        yrs=(idx[m].max()-idx[m].min()).days/365.25
        t=x[(x.dt>=idx[m].min())&(x.dt<=idx[m].max())]['r'].values
        w=t>0 if len(t) else np.array([False])
        o.append(f'{nm} {B[m].sum()/yrs*100:>+6.2f}%({len(t):>3}笔 胜{w.mean()*100:>3.0f}% '
                 f'单笔{(t.mean()*1e4 if len(t) else 0):>+6.1f}bp)')
    return f'  {lab:<30}' + ' | '.join(o)

P('='*146); P('★ pp≥0.90 前提 + 噪声突破早进场 (早盘三笔 10:00/10:29/11:29, S2口径, 双边20bp)')
for code in KEY:
    d1,s=sim(code,'C1'); d2,_=sim(code,'C2'); d3,_=sim(code,'C3')
    idx=s.index[s.index>=TR0]
    npp=(s['pp']>=0.90).reindex(idx).fillna(False)
    nred=(s['red']>0).reindex(idx).fillna(False)
    P(); P(f'── {code}   pp≥0.90 的日子 {int(npp.sum())} 天; 其中最终 red>0 的 {int((npp&nred).sum())} 天 '
      f'({(npp&nred).sum()/max(npp.sum(),1)*100:.0f}%) ⇒ ★另有 {int((npp&~nred).sum())} 天 v11.9 空仓')
    P(rep(d1,idx,'C1 pp≥0.90 ∧ 突破 (用户提法)'))
    P(rep(d2,idx,'C2 pp≥0.90 无突破要求'))
    P(rep(d3,idx,'C3 突破 无pp要求(原版)'))
    P('  ── 仅 red=0 的日子(v11.9空仓 ⇒ 纯增量) ──')
    P(rep(d1,idx,'C1 纯增量',onlynonred=True))
    P(rep(d2,idx,'C2 纯增量',onlynonred=True))
    P(rep(d3,idx,'C3 纯增量',onlynonred=True))
open(os.path.join(PROJ,'output','beta','pp_gated.txt'),'w').write('\n'.join(lines))
print('saved')
