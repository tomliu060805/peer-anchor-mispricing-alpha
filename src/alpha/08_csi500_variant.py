# -*- coding: utf-8 -*-
"""v6: 中证500域内版. 网络在当日成分内构建(残差corr top-5), 信号只出成分股,
基准=成分等权. 合成信号 = rank(peer_gap_K5)+rank(zspread). dev/val 同前, test锁定."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, INDEX_ROOT
import os, json, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
CH=PROJ+'/charts/v6_csi500_domain'
os.makedirs(CH,exist_ok=True)
W,REBUILD,K,MOM,HOLD,COST=120,21,5,20,5,0.00125
DEV=('2015-06-01','2022-12-31'); VAL=('2023-01-01','2024-08-16'); END=np.datetime64('2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
code_ix={c:i for i,c in enumerate(codes)}
dnum=dates.astype('datetime64[D]')

# 成分宽表
mem=np.zeros((T,N),bool)
mp=INDEX_ROOT+'/weight/csi_500'
files={f[:10]:f for f in os.listdir(mp)}
for t in range(T):
    f=files.get(str(dates[t]))
    if f is None: continue
    d=pd.read_parquet(f'{mp}/{f}',columns=['stock_code'])
    ix=[code_ix[c] for c in d['stock_code'] if c in code_ix]
    mem[t,ix]=True
print('membership days:',int(mem.any(1).sum()),flush=True)

ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
fwd=np.full((T,N),np.nan,np.float32)
fwd[:T-HOLD]=(np.exp(logc[HOLD:]-logc[:-HOLD])-1).astype(np.float32)

start=np.searchsorted(dnum,np.datetime64('2014-08-01'))+W
rebuilds=list(range(start,T,REBUILD))
nets={}
for t1 in rebuilds:
    members=np.where(mem[t1-1])[0]
    if len(members)<300: continue
    Rw=ret[t1-W:t1][:,members]
    ok=(~np.isnan(Rw)).sum(0)>=110
    members=members[ok]; Rw=Rw[:,ok]
    X=np.nan_to_num(Rw,nan=0.0).astype(np.float64)
    mkt=X.mean(1,keepdims=True)
    beta=(X*mkt).sum(0)/np.maximum((mkt*mkt).sum(),1e-12)
    Xr=X-mkt@beta[None,:]
    dstr=str(dates[t1-1]); mkeys=[m for m in ind_months if m<=dstr]
    imap=ind_map[mkeys[-1]] if mkeys else {}
    inds=np.array([imap.get(codes[i],'') for i in members])
    for u in set(inds):
        sel=inds==u
        if sel.sum()>1: Xr[:,sel]-=Xr[:,sel].mean(1,keepdims=True)
    sd=Xr.std(0)+1e-12; Z=(Xr-Xr.mean(0))/sd
    C=(Z.T@Z/len(Z)).astype(np.float32); np.fill_diagonal(C,-9)
    n=len(members)
    part=np.argpartition(-C,K,axis=1)[:,:K]
    rows=np.arange(n)[:,None]; vals=C[rows,part]
    order=np.argsort(-vals,axis=1)
    nets[t1]=(members,part[rows,order],vals[rows,order])
rb=np.array(sorted(nets.keys()))
print('networks:',len(rb),flush=True)

sig_days=np.arange(rb[0]+1,T-1,HOLD)
S_pg=np.full((len(sig_days),N),np.nan,np.float32)
S_zs=np.full((len(sig_days),N),np.nan,np.float32)
for si,t in enumerate(sig_days):
    t1=rb[np.searchsorted(rb,t,side='right')-1]
    members,nb,wv=nets[t1]
    m=mom[t]
    pm_v=m[members[nb]]
    w=np.maximum(wv,0)
    msk=~np.isnan(pm_v); w=w*msk
    sw=w.sum(1)
    agg=(np.nan_to_num(pm_v)*w).sum(1)/np.maximum(sw,1e-9)
    ok=sw>1e-9
    S_pg[si,members[ok]]=agg[ok]-m[members[ok]]
    # zspread
    t0=t1-W
    base=logc[t0-1] if t0>0 else np.zeros(N)
    Pw=np.exp(logc[t0:t1]-base)
    Pt=np.exp(logc[t]-base)
    si_p=Pw[:,members]; sj_p=Pw[:,members[nb].ravel()].reshape(W,len(members),K)
    s_tr=si_p[:,:,None]-sj_p
    mu=s_tr.mean(0); sd2=s_tr.std(0)+1e-9
    d=(s_tr**2).sum(0); d=d-d.min(1,keepdims=True)
    wsm=np.exp(-d); wsm=wsm/np.maximum(wsm.sum(1,keepdims=True),1e-12)
    s_t=Pt[members][:,None]-Pt[members[nb]]
    z=(s_t-mu)/sd2
    S_zs[si,members]=-np.nansum(z*wsm,1)

def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
COMBO=np.full_like(S_pg,np.nan)
for si in range(len(sig_days)):
    a,b2=rank_xs(S_pg[si]),rank_xs(S_zs[si])
    stk=np.stack([a,b2]); COMBO[si]=np.where(np.all(np.isnan(stk),0),np.nan,np.nanmean(stk,0))
S_rev=np.full_like(S_pg,np.nan)
for si,t in enumerate(sig_days):
    v=mem[t]&tradable[t]; S_rev[si,v]=-mom[t,v]

def seg_mask(t,a,b): return (dnum[t]>=np.datetime64(a))and(dnum[t]<=np.datetime64(b))
def evaluate(S,name):
    res={}; ann=252/HOLD
    for segn,(a,b) in [('dev',DEV),('val',VAL)]:
        ics=[];ls=[];te=[];prev=None
        for si,t in enumerate(sig_days):
            if not seg_mask(t,a,b): continue
            s=S[si].copy(); s[~(tradable[t]&mem[t])]=np.nan; f=fwd[t]
            m=~np.isnan(s)&~np.isnan(f)
            if m.sum()<200: prev=None; continue
            sv,fv=s[m],f[m]; ids=np.where(m)[0]
            xr=np.argsort(np.argsort(sv)).astype(float); yr=np.argsort(np.argsort(fv)).astype(float)
            xr=(xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
            ics.append(float((xr*yr).mean()))
            q=np.argsort(np.argsort(sv))/len(sv)
            top=set(ids[q>=0.9])
            to=1-len(top&prev)/max(len(top),1) if prev else 1.0
            rt=fv[q>=0.9].mean(); rb2=fv[q<0.1].mean(); ru=fv.mean()
            ls.append(rt-rb2-to*2*COST*2); te.append((1+rt-to*2*COST)/(1+ru)-1)
            prev=top
        ics,lsv,tev=map(np.array,(ics,ls,te))
        r=lsv
        res[segn]={'IC':round(float(ics.mean()),4),'ICIR':round(float(ics.mean()/(ics.std()+1e-12)),3),
          'LS_Sharpe':round(float(lsv.mean()/(lsv.std()+1e-12)*np.sqrt(ann)),2),
          'TopEx_ann':round(float((np.prod(1+tev))**(ann/len(tev))-1),4),
          'TopEx_Sharpe':round(float(tev.mean()/(tev.std()+1e-12)*np.sqrt(ann)),2),
          'n':int(len(r)),'win_te':round(float((tev>0).mean()),3),'avg_te_bp':round(float(tev.mean()*1e4),1),
          'pf_te':round(float(tev[tev>0].sum()/max(-tev[tev<0].sum(),1e-9)),2)}
    print(name,json.dumps(res),flush=True)
    return res

res={}
res['csi500_combo']=evaluate(COMBO,'csi500_combo')
res['csi500_peer_gap']=evaluate(S_pg,'csi500_peer_gap')
res['csi500_zspread']=evaluate(S_zs,'csi500_zspread')
res['csi500_rev']=evaluate(S_rev,'csi500_rev')

# 超额净值图 (combo vs rev)
def te_series(S):
    ds=[];te=[];prev=None
    for si,t in enumerate(sig_days):
        if dnum[t]>END or t+HOLD>=T: continue
        s=S[si].copy(); s[~(tradable[t]&mem[t])]=np.nan; f=fwd[t]
        m=~np.isnan(s)&~np.isnan(f)
        if m.sum()<200: prev=None; continue
        sv,fv=s[m],f[m]; ids=np.where(m)[0]
        q=np.argsort(np.argsort(sv))/len(sv)
        top=set(ids[q>=0.9])
        to=1-len(top&prev)/max(len(top),1) if prev else 1.0
        rt=fv[q>=0.9].mean()-to*2*COST; ru=fv.mean()
        ds.append(dnum[t]); te.append((1+rt)/(1+ru)-1); prev=top
    return np.array(ds),np.array(te)
d1,e1=te_series(COMBO); d2,e2=te_series(S_rev)
years=sorted(set(str(d)[:4] for d in d1))
yr={y:(float(np.prod(1+e1[[str(x)[:4]==y for x in d1]])-1),
       float(np.prod(1+e2[[str(x)[:4]==y for x in d2]])-1)) for y in years}
res['yearly_combo_vs_rev']={y:[round(v[0],4),round(v[1],4)] for y,v in yr.items()}
json.dump(res,open(f'{OUT}/metrics_v6_csi500.json','w'),ensure_ascii=False,indent=1)

fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,2]})
ax=axes[0]
ax.plot(d1,np.cumprod(1+e1),lw=1.6,color='#d62728',label='联动合成(500域内) Top10%=50只 超额净值(费后)')
ax.plot(d2,np.cumprod(1+e2),lw=1.2,color='#7f7f7f',ls='--',label='纯反转(500域内) 超额净值(费后)')
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.set_title('中证500域内: 多头Top10% 相对成分等权 复利超额净值 (周频, 单边12.5bp)')
ax.legend(); ax.grid(alpha=.3)
ax=axes[1]
x=np.arange(len(years)); wd=0.38
ax.bar(x-wd/2,[yr[y][0] for y in years],wd,color='#d62728',label='联动合成')
ax.bar(x+wd/2,[yr[y][1] for y in years],wd,color='#7f7f7f',label='纯反转')
for i,y in enumerate(years): ax.text(i-wd/2,yr[y][0]+(0.004 if yr[y][0]>=0 else -0.012),f'{yr[y][0]*100:.1f}',ha='center',fontsize=8)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额(%, 复利, 费后)'); ax.legend(); ax.grid(alpha=.3,axis='y')
plt.tight_layout(); plt.savefig(f'{CH}/csi500_domain_excess_nav.png',dpi=130); print('chart saved')
