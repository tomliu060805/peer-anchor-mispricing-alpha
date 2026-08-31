"""★修正 + 前视审计 + 决定性对照。

A. 前视/数据审计
   A1 指数的 money/volume 是否真有值(否则 VWAP 是垃圾)
   A2 VWAP 逐点重放: 抽样(日,bar), 只用截断到该 bar 的数据重算, 与向量化版本对拍
   A3 σ(T) 逐点重放: 抽样日期, 只用该日之前的数据重算
B. 修正: ★原实现 max(dn, vw[j]) 未检查 vw[j] 是否 NaN —— Python 的 max(数,nan) 行为依赖参数顺序,
   会静默产生错误止损位。修正为: VWAP 缺失时退回纯边界止损。
C. 三个决定性对照(全部 1bp 与 20bp 两档)
   C1 多空腿分解: 空头腿单独是否赚钱
   C2 日内漂移基线: 每日 09:31 开盘买入 / 14:57 卖出, 无条件
   C3 ★零信息对照: 保持完全相同的入场时点/持有时长/在场频率, 仅【随机化方向】, 200 个种子
"""
import os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
END=pd.Timestamp('2024-07-31'); DEC=[1029,1129,1359]; CLOSE_HM=1457; NDAY=14; DPY=244
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
df['dt']=pd.to_datetime(df['date']); df=df[df['dt']<=END]
lines=[]
def P(s=''):
    print(s); lines.append(s)

P('='*136)
P('★ A. 前视与数据审计')
P()
P('【A1 指数 money/volume 可用性(VWAP 的原料)】')
for c in ['SSE50','CSI300','CSI500','CSI1000','CNI2000']:
    g=df[df['code']==c]
    P(f'  {c:<9} money>0 占比 {(g["money"]>0).mean()*100:5.1f}%  volume>0 占比 {(g["volume"]>0).mean()*100:5.1f}%'
      f'  money中位 {g["money"].median():.3e}')

def panels(g):
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    pm=g.pivot_table(index='date',columns='hm',values='money')
    pv=g.pivot_table(index='date',columns='hm',values='volume')
    hms=np.array(sorted(pc.columns))
    return pc,po,pm,pv,hms

g5=df[df['code']=='CSI500']
pc,po,pm,pv,hms=panels(g5)
vwap_vec=(pm[hms].cumsum(axis=1)/pv[hms].cumsum(axis=1).replace(0,np.nan))
rng=np.random.RandomState(0)
P()
P('【A2 VWAP 逐点重放(CSI500, 抽样 300 个 (日,bar))】')
di=rng.choice(len(pc.index),300); bj=rng.choice(len(hms),300)
d_=[]
for a,b in zip(di,bj):
    m=pm.iloc[a,:b+1].values; v=pv.iloc[a,:b+1].values
    sv=np.nansum(v)
    ref=np.nansum(m)/sv if sv>0 else np.nan
    got=vwap_vec.iloc[a,b]
    if np.isfinite(ref) and np.isfinite(got): d_.append(abs(ref-got))
P(f'  最大绝对差 {max(d_) if d_ else float("nan"):.3e}  ({"PASS 无前视且实现一致" if d_ and max(d_)<1e-6 else "★FAIL"})')

P()
P('【A3 σ(T) 逐点重放(CSI500, 抽样 200 日 × T=1029)】')
dopen=po[hms[0]]
mv=(pc[1029]/dopen-1).abs()
sig_vec=mv.rolling(NDAY,min_periods=NDAY).mean().shift(1)
idx=np.arange(NDAY+1,len(pc.index)); pick=rng.choice(idx,200,replace=False)
d2=[]
for a in pick:
    ref=mv.iloc[a-NDAY:a].mean()          # 仅用该日之前 14 日
    got=sig_vec.iloc[a]
    if np.isfinite(ref) and np.isfinite(got): d2.append(abs(ref-got))
P(f'  最大绝对差 {max(d2) if d2 else float("nan"):.3e}  ({"PASS 窗口严格排除当日" if d2 and max(d2)<1e-12 else "★FAIL"})')

# ───────────── 修正后的策略 ─────────────
def run(g, fee_bp, vwap_stop=False, side_override=None, drift_only=False):
    """side_override: dict {(di, T): side} 用于零信息对照(强制方向)。
       drift_only: 每日09:31开盘买入/14:57卖出。"""
    fee=fee_bp*1e-4
    pc,po,pm,pv,hms=panels(g)
    hp={h:i for i,h in enumerate(hms)}; dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1)
         for T in DEC if T in pc.columns}
    vw_all=(pm[hms].cumsum(axis=1)/pv[hms].cumsum(axis=1).replace(0,np.nan)).values
    C=pc.values; O=po.values
    ic=np.where(hms<=CLOSE_HM)[0][-1]
    rets=[]; trades=[]; events=[]
    for di in range(len(pc.index)):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): rets.append(0.0); continue
        rc,ro,vw=C[di],O[di],vw_all[di]
        if drift_only:
            ep,xp=ro[0],rc[ic]
            r=(xp/ep-1)-2*fee if np.isfinite(ep) and np.isfinite(xp) else 0.0
            rets.append(r);  trades.append(r) if r else None; continue
        day=0.0; held=False
        for T in DEC:
            if held or T not in sig or T not in hp: continue
            s=sig[T].iloc[di]
            if not np.isfinite(s): continue
            iT=hp[T]; cT=rc[iT]
            if not np.isfinite(cT): continue
            up,dn=o0*(1+s), o0*(1-s)
            side=1 if cT>up else (-1 if cT<dn else 0)
            if side==0: continue
            if side_override is not None:
                side=side_override.get((di,T), side)
            nxt=iT+1
            if nxt>=len(hms) or hms[nxt]>CLOSE_HM: continue
            ep=ro[nxt]
            if not np.isfinite(ep): continue
            xp=None
            for j in range(nxt,len(hms)):
                if hms[j]>CLOSE_HM: break
                cj=rc[j]
                if not np.isfinite(cj): continue
                if vwap_stop and np.isfinite(vw[j]):        # ★修正: NaN 时退回纯边界
                    st=max(dn,vw[j]) if side>0 else min(up,vw[j])
                else:
                    st=dn if side>0 else up
                if ((cj<st) if side>0 else (cj>st)) and j+1<len(hms) and hms[j+1]<=CLOSE_HM:
                    xp=ro[j+1]; break
            if xp is None: xp=rc[ic]
            if np.isfinite(xp):
                r=side*(xp/ep-1)-2*fee
                day+=r; trades.append(r); events.append((di,T,side,r)); held=True
        rets.append(day)
    return pd.Series(rets,index=pd.to_datetime(pc.index)), np.array(trades), events

def stat(r,tr,lab):
    ann=r.mean()*DPY*100; vol=r.std()*np.sqrt(DPY)*100
    sh=ann/vol if vol>0 else np.nan
    eq=(1+r).cumprod(); mdd=((eq/eq.cummax())-1).min()*100
    w=tr>0 if len(tr) else np.array([False])
    return (f'{lab:<26} 年化{ann:>+7.2f}% Sharpe{sh:>6.2f} MDD{mdd:>7.2f}% '
            f'在场{(r!=0).mean()*100:>4.0f}% | {len(tr):>5}笔 胜{w.mean()*100:>3.0f}% 单笔{tr.mean()*1e4 if len(tr) else 0:>+6.1f}bp')

CODES=['CSI500','CSI1000','CNI2000']
P(); P('='*136)
P('★ B. 修正 NaN 止损前后对比 (改进版, 单边1bp)')
for c in CODES:
    g=df[df['code']==c]
    r,tr,_=run(g,1.0,vwap_stop=True)
    P('  ' + stat(r,tr,f'{c} 修正后改进版'))
P('  (原实现在 VWAP 为 NaN 时 max(dn,nan) 行为不定; 修正为退回纯边界止损)')

P(); P('='*136)
P('★ C. 三个决定性对照 (基础版, 无 VWAP)')
for fee_bp,flab in [(1.0,'单边1bp'),(10.0,'单边10bp=双边20bp')]:
    P(); P(f'── {flab} ──')
    for c in CODES:
        g=df[df['code']==c]
        r,tr,ev=run(g,fee_bp)
        P('  ' + stat(r,tr,f'{c} 策略'))
        # C1 多空腿
        ev=np.array([(e[2],e[3]) for e in ev])
        for sd,nm in [(1,'多头腿'),(-1,'空头腿')]:
            m=ev[:,0]==sd
            if m.sum():
                t=ev[m,1]
                P(f'      {nm}: {m.sum():>5}笔 胜{(t>0).mean()*100:>3.0f}% 单笔{t.mean()*1e4:>+6.1f}bp 合计{t.sum()*100:>+7.1f}%')
        # C2 漂移基线
        rd,td,_=run(g,fee_bp,drift_only=True)
        P('      ' + stat(rd,td,'日内漂移基线(9:31买/14:57卖)'))
    # C3 零信息对照(仅 1bp 档跑, 200 种子)
    if fee_bp==1.0:
        P()
        P('  ★C3 零信息对照: 入场时点/持有完全相同, 仅随机化方向(200种子)')
        for c in CODES:
            g=df[df['code']==c]
            r0,tr0,ev0=run(g,1.0)
            real=r0.mean()*DPY*100
            keys=[(e[0],e[1]) for e in ev0]
            rs=np.random.RandomState(42); sims=[]
            for _ in range(200):
                ov={k:(1 if rs.rand()<0.5 else -1) for k in keys}
                rr,_,_=run(g,1.0,side_override=ov)
                sims.append(rr.mean()*DPY*100)
            sims=np.array(sims)
            pct=(sims<real).mean()*100
            P(f'    {c:<9} 真实年化{real:>+7.2f}%  随机方向分布: 均值{sims.mean():>+7.2f}% 标准差{sims.std():>5.2f}% '
              f'95分位{np.percentile(sims,95):>+7.2f}%  ⇒ 真实位于{pct:>5.1f}分位')
open(os.path.join(PROJ,'output','beta','audit_and_fix.txt'),'w').write('\n'.join(lines))
print('\nsaved output/audit_and_fix.txt')
