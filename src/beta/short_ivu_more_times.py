"""★两项扩展: (1) 决策时点扩到 9:45/10:00/10:29/11:29/13:59; (2) IVU 状态门加到【空头腿】。
★前视改造: IVU30(T)=money(09:31→10:00)/money(09:31→T) 仅 T>=1000 可用;
           IVU5(T) =money(09:31→09:35)/money(09:31→T) 任意 T>935 可用(为 9:45 而设)。
   分位由过去59日同时刻决定(当日仅作被比较对象), 无前视。
判据: 训验双正且验证段单笔改善; 测试段2024-08起封存。
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); CLOSE_HM=1457; NDAY=14; FEE=10e-4
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
DEC5=[945,1000,1029,1129,1359]
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)
CACHE={}
def build(code,dec):
    key=(code,tuple(dec))
    if key in CACHE: return CACHE[key]
    g=_df[_df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    pm=g.pivot_table(index='date',columns='hm',values='money')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]; cum=pm[hms].cumsum(axis=1)
    n30=cum.iloc[:,hp[1000]] if 1000 in hp else None
    n5 =cum.iloc[:,hp[935]]  if 935  in hp else None
    pct=lambda s: s.rolling(60,min_periods=40).apply(lambda w:(w[:-1]<w[-1]).mean(),raw=True)
    ivu={}
    for T in dec:
        if T not in hp: continue
        den=cum.iloc[:,hp[T]].replace(0,np.nan)
        ivu[('30',T)]=pct(n30/den) if (n30 is not None and T>=1000) else None
        ivu[('5',T)] =pct(n5/den)  if (n5  is not None and T>935)  else None
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in dec if T in pc.columns}
    CACHE[key]=(pc,po,hms,hp,dopen,sig,ivu); return CACHE[key]

def run(code,k,dec,maxtr=3,sgate=None,thr=0.5,ver='30'):
    pc,po,hms,hp,dopen,sig,ivu=build(code,dec)
    C,O=pc.values,po.values; ic=np.where(hms<=CLOSE_HM)[0][-1]
    nxt=dopen.shift(-1).values; rows=[]
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        rc,ro=C[di],O[di]; ntr=0
        for T in dec:
            if ntr>=maxtr or T not in sig or T not in hp: continue
            s_=sig[T].iloc[di]; cT=rc[hp[T]]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT>=o0*(1-k*s_): continue            # 只做空头腿
            if sgate is not None:
                ser=ivu.get((ver,T))
                if ser is None: continue
                q=ser.iloc[di]
                if not np.isfinite(q): continue
                if sgate=='high' and q<=thr: continue
                if sgate=='low'  and q>=thr: continue
            j0=hp[T]+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            if not np.isfinite(ep): continue
            xp=nxt[di] if np.isfinite(nxt[di]) else rc[ic]
            if not np.isfinite(xp): continue
            rows.append((dstr,T,-(xp/ep-1)-2*FEE)); ntr+=1
    d=pd.DataFrame(rows,columns=['date','T','r']); d['dt']=pd.to_datetime(d['date']); return d

def rep(d,lab):
    o=[]
    for s,m in [('训',(d.dt>=TR0)&(d.dt<TRm)),('验',(d.dt>=TRm)&(d.dt<=END))]:
        t=d[m]['r'].values
        if not len(t): o.append(f'{s} 无'); continue
        w=t>0; pfn=-t[~w].sum()
        o.append(f'{s} {len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp '
                 f'PF{(t[w].sum()/pfn if pfn>0 else np.inf):>5.2f} 合计{t.sum()*100:>+6.1f}%')
    return f'  {lab:<28}' + ' | '.join(o)

CODES=['CSI500','CSI1000','CNI2000']
P('='*136); P('★ 扩展: +9:45/10:00 时点, IVU 门加到空头腿 (训/验决定, 测试段封存)')
P(); P('【E1 各决策时点单笔边际(k=2.0, 5时点, 每日至多5笔)】')
for c in CODES:
    d=run(c,2.0,DEC5,maxtr=5); row=[]
    for T in DEC5:
        x=d[d['T']==T]
        tr=x[(x.dt>=TR0)&(x.dt<TRm)]['r'].values; va=x[(x.dt>=TRm)&(x.dt<=END)]['r'].values
        row.append(f"{T}:训{tr.mean()*1e4 if len(tr) else float('nan'):>+5.1f}/验{va.mean()*1e4 if len(va) else float('nan'):>+5.1f}({len(x):>4})")
    P(f'  {c:<9}' + ' '.join(row))
P(); P('【E2 时点集合对比(k=2.0, 只空头)】')
for c in CODES:
    P(f'  ── {c}')
    for dec,nm in [([1029,1129,1359],'原3时点'),(DEC5,'5时点(+9:45,10:00)'),
                   ([945,1000,1029],'仅早盘3时点'),([1000,1029,1129,1359],'4时点(+10:00)')]:
        P(rep(run(c,2.0,dec,maxtr=len(dec)),nm))
P(); P('【E3 ★IVU 门加到空头腿(5时点)】')
for c in CODES:
    P(f'  ── {c}')
    for k in [2.0,2.5]:
        P(rep(run(c,k,DEC5,maxtr=5),f'k={k} 无门'))
        for ver in ['30','5']:
            for gt in ['high','low']:
                for th in [0.3,0.5,0.7]:
                    d=run(c,k,DEC5,maxtr=5,sgate=gt,thr=th,ver=ver)
                    if len(d)>40: P(rep(d,f'k={k} IVU{ver} {gt}>{th}'))
        P()
open(os.path.join(PROJ,'output','beta','short_ivu_more_times.txt'),'w').write('\n'.join(lines))
print('saved')
