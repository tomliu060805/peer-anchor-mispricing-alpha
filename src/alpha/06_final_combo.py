# -*- coding: utf-8 -*-
"""v4: 最终信号(peer_gap_K5 与 zspread 的合成) + 逐年IC + 净值曲线图."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f or 'NotoSerifCJK' in f:
        font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
CH=PROJ+'/charts/v4_signal_overview'
os.makedirs(CH,exist_ok=True)
W,MOM,HOLD,COST=120,20,5,0.00125
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
nw=np.load(f'{CACHE}/networks.npz'); rebuilds=nw['rebuilds']
nbr_resid,wgt_resid=nw['nbr_resid'],nw['wgt_resid']
z=np.load(f'{OUT}/signals_resid.npz'); sig_days=z['sig_days']; ZS=z['zspread']
zr=np.load(f'{OUT}/signals_random.npz'); PG_rand=zr['peer_gap']
def last_rebuild(t): return np.searchsorted(rebuilds,t,side='right')-1
fwd=np.full((T,N),np.nan,np.float32)
fwd[:T-HOLD]=(np.exp(logc[HOLD:]-logc[:-HOLD])-1).astype(np.float32)

# peer_gap K5
PG5=np.full((len(sig_days),N),np.nan,np.float32)
for si,t in enumerate(sig_days):
    b=last_rebuild(t); nbr=nbr_resid[b][:,:5]; wgt=np.maximum(wgt_resid[b][:,:5],0)
    m=mom[t]; pm=np.take(m,np.where(nbr>=0,nbr,0))
    mask=(nbr>=0)&~np.isnan(pm); w=wgt*mask
    sw=w.sum(1); ok=sw>1e-9
    agg=(np.where(mask,np.nan_to_num(pm),0)*w).sum(1)/np.maximum(sw,1e-9)
    rows=np.where(ok)[0]; PG5[si,rows]=agg[rows]-m[rows]
S_rev=np.stack([-mom[t] for t in sig_days])

def rank_xs(x):
    m=~np.isnan(x); r=np.full_like(x,np.nan,dtype=np.float32)
    if m.sum()<50: return r
    rr=np.argsort(np.argsort(x[m])).astype(np.float32)
    r[m]=(rr-rr.mean())/(rr.std()+1e-12); return r
COMBO=np.full_like(PG5,np.nan)
for si in range(len(sig_days)):
    a,b2=rank_xs(PG5[si]),rank_xs(ZS[si])
    COMBO[si]=np.nanmean(np.stack([a,b2]),0)

def seg_mask(t,seg): return (dnum[t]>=np.datetime64(seg[0]))and(dnum[t]<=np.datetime64(seg[1]))
def run_bt(S):
    """返回逐期 dict: date, ic, ls_net, top_ex_net"""
    out={'date':[],'ic':[],'ls':[],'te':[]}
    prev_top=None; ann=252/HOLD
    for si,t in enumerate(sig_days):
        if not (seg_mask(t,DEV) or seg_mask(t,VAL)): continue
        s=S[si].copy(); s[~tradable[t]]=np.nan; f=fwd[t]
        m=~np.isnan(s)&~np.isnan(f)
        if m.sum()<100: prev_top=None; continue
        sv,fv=s[m],f[m]; ids=np.where(m)[0]
        xr=np.argsort(np.argsort(sv)).astype(float); yr=np.argsort(np.argsort(fv)).astype(float)
        xr=(xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
        q=np.argsort(np.argsort(sv))/len(sv)
        top=set(ids[q>=0.9])
        to=1-len(top&prev_top)/max(len(top),1) if prev_top else 0.5
        rt=fv[q>=0.9].mean(); rb=fv[q<0.1].mean(); ru=fv.mean()
        out['date'].append(dnum[t]); out['ic'].append(float((xr*yr).mean()))
        out['ls'].append(rt-rb-to*2*COST*2); out['te'].append(rt-ru-to*2*COST)
        prev_top=top
    for k in ['ic','ls','te']: out[k]=np.array(out[k])
    out['date']=np.array(out['date'])
    return out

names={'纯反转 -mom20':S_rev,'随机锚 peer_gap':PG_rand,'联动锚 peer_gap_K5':PG5,'zspread(联动)':ZS,'合成 K5+zspread':COMBO}
bts={k:run_bt(v) for k,v in names.items()}

# 逐年 IC
years=sorted(set(str(d)[:4] for d in bts['联动锚 peer_gap_K5']['date']))
yearly={}
res={}
for nm,bt in bts.items():
    ys={}
    for y in years:
        m=np.array([str(d)[:4]==y for d in bt['date']])
        if m.sum()<5: continue
        ic=bt['ic'][m]; ys[y]={'IC':round(float(ic.mean()),4),'ICIR':round(float(ic.mean()/(ic.std()+1e-12)),2)}
    yearly[nm]=ys
    for segn,seg in [('dev',DEV),('val',VAL)]:
        m=np.array([(d>=np.datetime64(seg[0]))&(d<=np.datetime64(seg[1])) for d in bt['date']])
        ic,ls,te=bt['ic'][m],bt['ls'][m],bt['te'][m]
        ann=252/HOLD
        r=ls
        res[f'{nm}:{segn}']={'IC':round(float(ic.mean()),4),'ICIR':round(float(ic.mean()/(ic.std()+1e-12)),3),
            'LS_ann_net':round(float(ls.mean()*ann),3),'Sharpe':round(float(ls.mean()/(ls.std()+1e-12)*np.sqrt(ann)),2),
            'TopEx_ann_net':round(float(te.mean()*ann),3),
            'n':int(len(r)),'win':round(float((r>0).mean()),3),'avg_bp':round(float(r.mean()*1e4),1),
            'pf':round(float(r[r>0].sum()/max(-r[r<0].sum(),1e-9)),2)}
json.dump({'summary':res,'yearly':yearly},open(f'{OUT}/metrics_v4_signal_overview.json','w'),ensure_ascii=False,indent=1)
for k,v in res.items(): print(k,v)

# ---- 图 ----
fig,axes=plt.subplots(2,2,figsize=(15,10))
ax=axes[0,0]
for nm in ['纯反转 -mom20','随机锚 peer_gap','联动锚 peer_gap_K5','合成 K5+zspread']:
    bt=bts[nm]; ax.plot(bt['date'],np.cumsum(bt['ls']),label=nm,lw=1.2)
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls='--',lw=.8)
ax.set_title('费后多空累计收益 (十分组, 周频, 单边12.5bp)'); ax.legend(fontsize=9); ax.grid(alpha=.3)
ax=axes[0,1]
for nm in ['纯反转 -mom20','联动锚 peer_gap_K5','合成 K5+zspread']:
    bt=bts[nm]; ax.plot(bt['date'],np.cumsum(bt['te']),label=nm,lw=1.2)
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls='--',lw=.8)
ax.set_title('费后多头(Top10%)相对全域超额累计'); ax.legend(fontsize=9); ax.grid(alpha=.3)
ax=axes[1,0]
labels=['市场锚','随机锚','K10','K5','K3','互选']
dev_icir=[.412,.427,.575,.596,.615,.594]; val_icir=[.305,.314,.44,.459,.462,.443]
x=np.arange(len(labels)); wdt=0.35
ax.bar(x-wdt/2,dev_icir,wdt,label='dev'); ax.bar(x+wdt/2,val_icir,wdt,label='val')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title('锚质量单调性: peer_gap ICIR 随联动强度上升')
ax.legend(); ax.grid(alpha=.3,axis='y')
ax=axes[1,1]
nm='合成 K5+zspread'; bt=bts[nm]
ys=[y for y in years]; icv=[]
for y in ys:
    m=np.array([str(d)[:4]==y for d in bt['date']]); icv.append(bt['ic'][m].mean() if m.sum()>0 else np.nan)
ax.bar(ys,icv); ax.set_title(f'{nm} 逐年IC'); ax.grid(alpha=.3,axis='y'); ax.tick_params(axis='x',rotation=45)
plt.tight_layout(); plt.savefig(f'{CH}/network_signal_overview.png',dpi=130); print('chart saved')
