"""★把突破信号当作 v11.9 的条件筛选器(而非独立book)。
时序合法性: 突破在 10:29/11:29/13:59 判定, 全部早于 v11.9 的 14:00 决策。
分组: v11.9 触发日中, 当日 14:00 前是否出现【向下突破】(以及【任意突破】作对照)。
指标: 项目铁律(触发数/胜率/单笔均/盈亏比/盈利因子), 训/验分列; 测试段2024-08起封存。
经济检验: 只在有向下突破确认的 red 日交易 v11.9 —— 书变好还是变差?
"""
import os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); DEC=[1029,1129,1359]; NDAY=14; DPY=244
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
KEY={'CSI500':'500','CSI1000':'1000','CNI2000':'2000'}
FEE=10e-4
_df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
_df['dt']=pd.to_datetime(_df['date']); _df=_df[_df['dt']<=END]
PAN=pd.read_parquet(EXT_REGIME+'/data/panel_official_idx.parquet')
PAN['date']=pd.to_datetime(PAN['date'].astype(str).str[:10])
lines=[]
def P(s=''):
    print(s); lines.append(s)

def breakouts(code,k):
    """返回 {date: set(方向)} —— 当日 14:00 前出现过的突破方向。"""
    g=_df[_df['code']==code]
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC if T in pc.columns}
    out={}
    for di,dstr in enumerate(pc.index):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): continue
        st=set()
        for T in DEC:
            if T not in sig or T not in pc.columns: continue
            s_=sig[T].iloc[di]; cT=pc[T].iloc[di]
            if not (np.isfinite(s_) and np.isfinite(cT)): continue
            if cT> o0*(1+k*s_): st.add(1)
            elif cT< o0*(1-k*s_): st.add(-1)
        out[pd.Timestamp(dstr)]=st
    return out

def v119_trades(key):
    s=pd.read_parquet(f'{EXT_REGIME}/output/series_{key}.parquet')
    s['date']=pd.to_datetime(s['date']); s=s.set_index('date')
    pan=PAN[PAN['key']==key].set_index('date')
    gap_next=pan['gap_idx'].shift(-1).reindex(s.index).fillna(0)
    on=s['red']*s['osc']
    pnl=s['red']*(-s['r14'])-2*s['red']*FEE+on*(-gap_next)
    d=pd.DataFrame({'pnl':pnl,'red':s['red'],'bench':s['bench'],'strat':s['strat']},index=s.index)
    return d[(d.red>0)&(d.index>=TR0)&(d.index<=END)], s

def tm(t):
    if not len(t): return '  无样本'
    w=t>0; aw=t[w].mean() if w.any() else 0; al=t[~w].mean() if (~w).any() else 0
    pfn=-t[~w].sum()
    return (f'{len(t):>4}笔 胜{w.mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp '
            f'盈亏比{abs(aw/al) if al else np.inf:>5.2f} PF{(t[w].sum()/pfn if pfn>0 else np.inf):>5.2f} '
            f'合计{t.sum()*100:>+6.1f}%')

P('='*140); P('★ 突破信号作为 v11.9 的条件筛选器 (训/验决定, 测试段2024-08起封存)')
P('  时序: 突破在 10:29/11:29/13:59 判定, 早于 v11.9 的 14:00 决策 ⇒ 合法 conditioner')
for code in ['CSI500','CSI1000','CNI2000']:
    key=KEY[code]
    tr,_=v119_trades(key)
    P(); P(f'── {code} (v11.9 触发日共 {len(tr)} 天)')
    for k in [1.5,2.0,2.5]:
        bo=breakouts(code,k)
        has_dn=np.array([(-1) in bo.get(d,set()) for d in tr.index])
        has_any=np.array([len(bo.get(d,set()))>0 for d in tr.index])
        for nm,mask in [('有向下突破',has_dn),('无向下突破',~has_dn)]:
            o=[]
            for lab,sm in [('训',(tr.index>=TR0)&(tr.index<TRm)),('验',(tr.index>=TRm)&(tr.index<=END))]:
                o.append(f'{lab} '+tm(tr['pnl'].values[mask&sm]))
            P(f'  k={k} {nm:<8} ' + ' | '.join(o))
        P(f'  k={k} {"覆盖率":<8} 有向下突破占 {has_dn.mean()*100:>4.1f}% · 有任意突破占 {has_any.mean()*100:>4.1f}%')
        P()

P('='*140); P('★ 经济检验: 只在【有向下突破确认】的 red 日交易 v11.9 (其余日子不动仓)')
for code in ['CSI500','CSI1000','CNI2000']:
    key=KEY[code]
    tr,s=v119_trades(key)
    pan=PAN[PAN['key']==key].set_index('date').reindex(s.index)
    er=pd.to_numeric(pan['er_idx'],errors='coerce').values
    r10=pd.to_numeric(pan['rest10'],errors='coerce').values
    gap=np.nan_to_num(pd.to_numeric(pan['gap_idx'],errors='coerce').values)
    full=(1+np.nan_to_num(er))*(1+np.nan_to_num(r10))-1
    r14=pd.to_numeric(s['r14'],errors='coerce').values
    idx=s.index
    P(); P(f'── {code}')
    for k in [None,1.5,2.0,2.5]:
        red=s['red'].copy()
        if k is not None:
            bo=breakouts(code,k)
            keep=np.array([(-1) in bo.get(d,set()) for d in idx])
            red=red.where(keep,0.0)
        rv=red.values; on=(red*s['osc']).values
        bench=(1+gap)*(1+np.nan_to_num(full))-1
        sess=np.nan_to_num(full).copy(); strat=np.zeros(len(rv))
        for t in range(len(rv)):
            if rv[t]>0: sess[t]=full[t]-rv[t]*r14[t]-2*rv[t]*FEE
            h=on[t-1] if t>0 and rv[t-1]>0 else 0.0
            strat[t]=(1+gap[t]*(1-h))*(1+sess[t])-1
        o=[]
        for lab,m in [('训',(idx>=TR0)&(idx<TRm)),('验',(idx>=TRm)&(idx<=END))]:
            ex=(np.prod(1+strat[m])-1)-(np.prod(1+bench[m])-1)
            nt=int((rv[m]>0).sum())
            o.append(f'{lab} 超额{ex*100:>+7.2f}% ({nt:>3}笔)')
        P(f'  {"原版v11.9" if k is None else f"仅k={k}向下突破日":<20}' + ' | '.join(o))
open(os.path.join(PROJ,'output','beta','as_conditioner.txt'),'w').write('\n'.join(lines))
print('\nsaved output/as_conditioner.txt')
