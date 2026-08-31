"""★冻结存证 + 测试段一次性开封 (2026-08-28)。
仓位口径按用户指定 = S2(每笔0.25单位, 同日最多4笔 ⇒ 峰值1x, 无杠杆)。
存证在开封前写入并计SHA; 测试段 2024-08-01 起, 此前从未触碰。
"""
import os, json, hashlib, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(os.path.dirname(HERE))
CLOSE_HM=1457; NDAY=14; FEE=10e-4; K=2.5; DEC=[1000,1029,1129,1359]; UNIT=0.25
TR0,TRm=pd.Timestamp('2016-01-01'),pd.Timestamp('2022-02-15')
VAL_END=pd.Timestamp('2024-07-31'); TE0=pd.Timestamp('2024-08-01')
df=pd.read_parquet(os.path.join(PROJ,'data','idx1m.parquet'))
df['dt']=pd.to_datetime(df['date'])
END=df['dt'].max()
lines=[]
def P(s=''):
    print(s); lines.append(s)

CFG={
 'version':'早盘噪声突破·空头腿 v1 (S2仓位)','frozen_date':'2026-08-28',
 'universe':['CSI500(000905)','CSI1000(000852)','CNI2000(399303)'],
 'signal':{'sigma':'过去14个交易日 |close_d(T)/open_d − 1| 的均值, shift(1)严格排除当日',
           'band':'今日开盘价 × (1 ± k·σ(T)), k=2.5',
           'decision_times':['10:00','10:29','11:29','13:59'],
           'rule':'close(T) < Lower(T) → 做空; 只做空头腿(多头腿已判负)'},
 'execution':{'entry':'判定bar的下一根bar开盘价','exit':'次日第一根bar开盘价(不留止损)',
              'note':'★出场必须精确落在开盘那一刻: 晚4分钟少2.1~8.1bp'},
 'sizing':{'scheme':'S2 每笔0.25单位, 同日最多4笔','peak_exposure':'1.0x','avg_exposure':'0.04~0.05',
           'note':'杠杆倍数留给组合层决定; S2/S3/S4 为同一策略的不同倍数, Sharpe相同'},
 'cost':'双边20bp(单边10bp), 所有数字已扣除',
 'split':{'train':'2016-01-01~2022-02-14','val':'2022-02-15~2024-07-31','test':'2024-08-01起'},
 'declared_trainval_S2':{'CSI500':'训+1.76/验+3.08 %/年, SR0.51',
                         'CSI1000':'训+3.51/验+4.33 %/年, SR0.76',
                         'CNI2000':'训+4.18/验+5.70 %/年, SR0.94'},
 'audits_passed':['σ逐点重放(最大差8e-10=浮点噪声)','决策成交时序4/4严格分离',
                  'shift(-1)仅用于出场价与目标, 未进入决策','09:31开盘价==前收比例0.00%',
                  '零信息对照(随机方向500种子) z=6.4~9.2','突破时恒做空基准: 训-0.7/验+4.8bp(策略+48.3/+28.8)',
                  '无条件做空基准: -10.5bp'],
 'known_costs':['九年唯一负年2021(-5.0%), 与v11.9弱年重合, 未治好2021',
                '在场时间仅9%~10%, 年化天花板受此约束',
                '与v11.9吃同一块结构性的钱: 非red日验证段为负, 作筛选器/并行附庸均判负',
                '★搜索偏差: 本线累计跑过数百个配置, 真实预期应打折'],
}
js=json.dumps(CFG,ensure_ascii=False,indent=1)
sha=hashlib.sha256(js.encode()).hexdigest()[:12]
open(os.path.join(PROJ,'config','config_breakout_short_v1.json'),'w',encoding='utf-8').write(js)
P('='*128); P(f'★ 存证已写入 frozen/config_breakout_short_v1.json   SHA {sha}')
P(f'  仓位=S2(每笔0.25单位,峰值1x) | 声明训验(S2): 500 +1.76/+3.08 · 1000 +3.51/+4.33 · 2000 +4.18/+5.70 %/年')
P(f'★ 测试段开封: 2024-08-01 ~ {END.date()} (此前从未触碰)')

def run(code):
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
    daily=(d.groupby('dt')['r'].sum()*UNIT).reindex(pd.to_datetime(pc.index)).fillna(0)
    expo=(d.groupby('dt')['r'].size()*UNIT).reindex(pd.to_datetime(pc.index)).fillna(0)
    return d, daily, expo

RES={}
for code in ['CSI500','CSI1000','CNI2000']:
    d,r,e=run(code); RES[code]=(d,r,e)
    P(); P(f'── {code}')
    for lab,a,b in [('训',TR0,TRm-pd.Timedelta(days=1)),('验',TRm,VAL_END),('★测',TE0,END)]:
        m=(r.index>=a)&(r.index<=b); rr=r[m]; ee=e[m]
        yrs=(b-a).days/365.25
        ann=rr.mean()*244*100; vol=rr.std()*np.sqrt(244)*100
        eq=(1+rr).cumprod(); mdd=((eq/eq.cummax())-1).min()*100
        t=d[(d.dt>=a)&(d.dt<=b)]['r'].values
        w=t>0; pfn=-t[~w].sum() if len(t) else 0
        P(f'  {lab:<3} 年化{ann:>+6.2f}% SR{(ann/vol if vol>0 else np.nan):>5.2f} MDD{mdd:>6.2f}% 均敞口{ee.mean():>5.3f} '
          f'| {len(t):>4}笔 胜{(w.mean()*100 if len(t) else 0):>3.0f}% 单笔{(t.mean()*1e4 if len(t) else 0):>+6.1f}bp '
          f'盈亏比{(abs(t[w].mean()/t[~w].mean()) if (w.any() and (~w).any()) else np.nan):>4.2f} '
          f'PF{(t[w].sum()/pfn if pfn>0 else np.inf):>5.2f}')
    W=(r.index>=TR0)
    yr=r[W].groupby(r[W].index.year).sum()*100
    P('      逐年% ' + ' '.join(f'{y}:{v:+.1f}' for y,v in yr.items()))
open(os.path.join(PROJ,'output','beta','freeze_and_open.txt'),'w').write('\n'.join(lines))
print(f'\nSHA {sha}')
