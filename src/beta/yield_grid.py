"""★red日让位 × k × 每日笔数 网格 (100核并行)。20bp; 训/验决定, 测试段2024-08起封存。
让位三档: none=不让位 | all=red日14:00全平 | long=red日14:00只平多头(保留与v11.9同向的空头)
★申报: 本轮为参数搜索, 共 3指数×4k×2笔数×3让位 = 72 配置。判据要求【台地】而非单点:
   同一让位档下, k 的相邻档位需训验双正连成片, 且三指数同向。
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

def load_red(k):
    s=pd.read_parquet(f'{EXT_REGIME}/output/series_{k}.parquet')
    s['date']=pd.to_datetime(s['date'])
    return set(s[s['red']>0]['date'].dt.strftime('%Y-%m-%d'))

def work(args):
    code,k,maxtr,ymode = args
    g=_df[_df['code']==code]
    red=load_red(KEY[code])
    fee=10e-4
    pc=g.pivot_table(index='date',columns='hm',values='close')
    po=g.pivot_table(index='date',columns='hm',values='open')
    hms=np.array(sorted(pc.columns)); hp={h:i for i,h in enumerate(hms)}
    dopen=po[hms[0]]
    sig={T:(pc[T]/dopen-1).abs().rolling(NDAY,min_periods=NDAY).mean().shift(1) for T in DEC if T in pc.columns}
    C,O=pc.values,po.values
    ic=np.where(hms<=CLOSE_HM)[0][-1]
    i14=hp.get(1400)
    nxt=dopen.shift(-1).values
    dates=list(pc.index)
    rets=[]; recs=[]
    for di,dstr in enumerate(dates):
        o0=dopen.iloc[di]
        if not np.isfinite(o0): rets.append(0.0); continue
        rc,ro=C[di],O[di]; day=0.0; ntr=0; isred=dstr in red
        for T in DEC:
            if ntr>=maxtr or T not in sig or T not in hp: continue
            s_=sig[T].iloc[di]
            if not np.isfinite(s_): continue
            iT=hp[T]; cT=rc[iT]
            if not np.isfinite(cT): continue
            up,dn=o0*(1+k*s_), o0*(1-k*s_)
            side=1 if cT>up else (-1 if cT<dn else 0)
            if side==0: continue
            j0=iT+1
            if j0>=len(hms) or hms[j0]>CLOSE_HM: continue
            ep=ro[j0]
            if not np.isfinite(ep): continue
            # 让位判定
            do_yield = isred and i14 is not None and hms[j0]<=1400 and (
                ymode=='all' or (ymode=='long' and side>0))
            if do_yield and np.isfinite(rc[i14]):
                xp=rc[i14]
            elif side<0:
                xp=nxt[di]
                if not np.isfinite(xp): xp=rc[ic]
            else:
                xp=rc[ic]
            if np.isfinite(xp):
                r=side*(xp/ep-1)-2*fee
                day+=r; ntr+=1; recs.append((dstr,side,r))
        rets.append(day)
    r=pd.Series(rets,index=pd.to_datetime(dates))
    d=pd.DataFrame(recs,columns=['date','side','r'])
    out={'code':code,'k':k,'maxtr':maxtr,'ymode':ymode}
    dd=pd.to_datetime(d['date']) if len(d) else pd.Series([],dtype='datetime64[ns]')
    for lab,m,dm in [('tr',(r.index>=TR0)&(r.index<TRm),(dd>=TR0)&(dd<TRm)),
                     ('va',(r.index>=TRm)&(r.index<=END),(dd>=TRm)&(dd<=END))]:
        rr=r[m]; ann=rr.mean()*DPY*100; vol=rr.std()*np.sqrt(DPY)*100
        t=d['r'].values[dm.values] if len(d) else np.array([])
        out[f'{lab}_ann']=ann; out[f'{lab}_sr']=ann/vol if vol>0 else np.nan
        out[f'{lab}_n']=len(t); out[f'{lab}_bp']=t.mean()*1e4 if len(t) else np.nan
        out[f'{lab}_win']=(t>0).mean()*100 if len(t) else np.nan
        pfp=t[t>0].sum() if len(t) else 0; pfn=-t[t<0].sum() if len(t) else 0
        out[f'{lab}_pf']=pfp/pfn if pfn>0 else np.nan
    return out

if __name__=='__main__':
    grid=[(c,k,m,y) for c in ['CSI500','CSI1000','CNI2000']
                     for k in [1.5,2.0,2.5,3.0] for m in [1,3] for y in ['none','all','long']]
    print(f'配置数 {len(grid)}, 100核并行')
    with ProcessPoolExecutor(max_workers=min(100,len(grid))) as ex:
        res=list(ex.map(work,grid))
    R=pd.DataFrame(res)
    R.to_csv(os.path.join(PROJ,'output','beta','yield_grid.csv'),index=False)
    L=[]
    def P(s=''):
        print(s); L.append(s)
    YN={'none':'不让位','all':'red日14:00全平','long':'red日14:00只平多头'}
    P('='*140); P('★ red日让位 × k × 每日笔数 网格 @双边20bp (无止损+空头隔夜; 训/验决定, 测试封存)')
    P('  ★申报: 72 个配置的参数搜索; 判据=台地(相邻k训验双正连成片)且三指数同向, 非单点最优')
    for y in ['none','long','all']:
        P(); P(f'── 让位档: {YN[y]}')
        for m in [1,3]:
            P(f'   每日最多{m}笔')
            for c in ['CSI500','CSI1000','CNI2000']:
                row=[]
                for k in [1.5,2.0,2.5,3.0]:
                    x=R[(R.code==c)&(R.k==k)&(R.maxtr==m)&(R.ymode==y)].iloc[0]
                    ok='✓' if (x.tr_ann>0 and x.va_ann>0) else ' '
                    row.append(f'k{k}:{x.tr_ann:>+6.1f}/{x.va_ann:>+6.1f}{ok}')
                P(f'     {c:<9}' + '  '.join(row))
    P(); P('★ 训验双正且验证单笔>5bp 的配置(按验证段年化排序):')
    good=R[(R.tr_ann>0)&(R.va_ann>0)&(R.va_bp>5)].sort_values('va_ann',ascending=False)
    for _,x in good.iterrows():
        P(f'  {x.code:<9} k={x.k} {x.maxtr}笔/日 {YN[x.ymode]:<18} '
          f'训{x.tr_ann:>+6.1f}%(SR{x.tr_sr:>4.2f}) 验{x.va_ann:>+6.1f}%(SR{x.va_sr:>4.2f}) | '
          f'验 {int(x.va_n):>4}笔 胜{x.va_win:>3.0f}% 单笔{x.va_bp:>+5.1f}bp PF{x.va_pf:>4.2f}')
    P(f'  → 共 {len(good)}/{len(R)} 个配置通过')
    open(os.path.join(PROJ,'output','beta','yield_grid.txt'),'w').write('\n'.join(L))
