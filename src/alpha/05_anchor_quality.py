# -*- coding: utf-8 -*-
"""v3: 联动强度是否单调改善下游信号 (锚质量检验).
比较(统一组合级回测): rev基准 / market_gap / peer_gap K10/K5强联动/互为topK / zspread corr过滤
+ 配对级: 按残差corr三分位的价差收敛检验."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, numpy as np

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,REBUILD,K,MOM,HOLD=120,21,10,20,5
COST=0.00125
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates=g['ret'],g['dates']; T,N=ret.shape
paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom=np.full((T,N),np.nan,np.float32); mom[MOM:]=(logc[MOM:]-logc[:-MOM]).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
nw=np.load(f'{CACHE}/networks.npz'); rebuilds=nw['rebuilds']; B=len(rebuilds)
nbr_resid,wgt_resid=nw['nbr_resid'],nw['wgt_resid']; valid_all=nw['valid']
sig_days=np.arange(W+REBUILD,T-1,HOLD)
def last_rebuild(t): return np.searchsorted(rebuilds,t,side='right')-1
fwd=np.full((T,N),np.nan,np.float32)
fwd[:T-HOLD]=(np.exp(logc[HOLD:]-logc[:-HOLD])-1).astype(np.float32)

def seg_mask(t,seg): return (dnum[t]>=np.datetime64(seg[0]))and(dnum[t]<=np.datetime64(seg[1]))
def evaluate(S,name):
    res={}
    ann=252/HOLD
    for segn,seg in [('dev',DEV),('val',VAL)]:
        ics=[];ls=[];te=[];turns=[];prev_top=None
        for si,t in enumerate(sig_days):
            if not seg_mask(t,seg): continue
            s=S[si].copy(); s[~tradable[t]]=np.nan; f=fwd[t]
            m=~np.isnan(s)&~np.isnan(f)
            if m.sum()<100: continue
            sv,fv=s[m],f[m]; ids=np.where(m)[0]
            xr=np.argsort(np.argsort(sv)).astype(float); yr=np.argsort(np.argsort(fv)).astype(float)
            xr=(xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
            ics.append(float((xr*yr).mean()))
            q=np.argsort(np.argsort(sv))/len(sv)
            top=set(ids[q>=0.9]); rt=fv[q>=0.9].mean(); rb=fv[q<0.1].mean(); ru=fv.mean()
            ls.append(rt-rb); te.append(rt-ru)
            if prev_top: turns.append(1-len(top&prev_top)/max(len(top),1))
            prev_top=top
        ics,lsv,tev=map(np.array,(ics,ls,te))
        to=np.mean(turns) if turns else np.nan
        ls_net=lsv-(to*2*COST*2 if to==to else 0); te_net=tev-(to*2*COST if to==to else 0)
        r=ls_net
        win=(r>0).mean(); pf=r[r>0].sum()/max(-r[r<0].sum(),1e-9)
        res[segn]={'IC':round(float(ics.mean()),4),'ICIR':round(float(ics.mean()/(ics.std()+1e-12)),3),
          'LS_ann_net':round(float(ls_net.mean()*ann),3),'LS_sharpe_net':round(float(ls_net.mean()/(ls_net.std()+1e-12)*np.sqrt(ann)),2),
          'TopEx_ann_net':round(float(te_net.mean()*ann),3),'turnover':round(float(to),3),
          'n':int(len(r)),'win':round(float(win),3),'avg_bp':round(float(r.mean()*1e4),1),'pf':round(float(pf),2)}
    print(name,json.dumps(res),flush=True)
    return res

def peer_gap_sig(kk=None,mutual=False,wpow=1.0):
    S=np.full((len(sig_days),N),np.nan,np.float32)
    for si,t in enumerate(sig_days):
        b=last_rebuild(t); nbr=nbr_resid[b].copy(); wgt=np.maximum(wgt_resid[b],0).copy()
        if kk: nbr=nbr[:,:kk]; wgt=wgt[:,:kk]
        if mutual:
            # 互为topK: j 的邻居表里也有 i
            nb_full=nbr_resid[b]
            for col in range(nbr.shape[1]):
                j=nbr[:,col]
                ok=j>=0
                mut=np.zeros(N,bool)
                rows=np.where(ok)[0]
                mut[rows]=(nb_full[j[rows]]==rows[:,None]).any(1)
                wgt[~mut,col]=0
        w=wgt**wpow*(nbr>=0)
        m=mom[t]
        pm=np.take(m,np.where(nbr>=0,nbr,0))
        mask=(nbr>=0)&~np.isnan(pm); w=w*mask
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.where(mask,np.nan_to_num(pm),0)*w).sum(1)/np.maximum(sw,1e-9)
        rows=np.where(ok)[0]
        S[si,rows]=agg[rows]-m[rows]
    return S

# market_gap: 市场等权动量 - own
S_mg=np.full((len(sig_days),N),np.nan,np.float32)
for si,t in enumerate(sig_days):
    b=last_rebuild(t); v=valid_all[b]
    m=mom[t]; mk=np.nanmean(np.where(v,m,np.nan))
    S_mg[si,v]=mk-m[v]
S_rev=np.full((len(sig_days),N),np.nan,np.float32)
for si,t in enumerate(sig_days): S_rev[si]=-mom[t]

res={}
res['rev']=evaluate(S_rev,'rev')
res['market_gap']=evaluate(S_mg,'market_gap')
res['peer_gap_K10']=evaluate(peer_gap_sig(),'peer_gap_K10')
res['peer_gap_K5']=evaluate(peer_gap_sig(kk=5),'peer_gap_K5')
res['peer_gap_K3']=evaluate(peer_gap_sig(kk=3),'peer_gap_K3')
res['peer_gap_mutual']=evaluate(peer_gap_sig(mutual=True),'peer_gap_mutual')
res['peer_gap_w2']=evaluate(peer_gap_sig(wpow=2.0),'peer_gap_w2')

# ---- 配对级收敛: 残差corr三分位 ----
# 检验: 每次rebuild的边(i,topK邻居), 信号 = -z_ij(t1), 未来HOLD日 (fwd_i - fwd_j) 与信号同号的程度
# 按 corr 三分位分组 -> 若强联动配对收敛更强, 应单调
pair_out={}
for segn in ['dev','val']: pair_out[segn]={0:[],1:[],2:[]}
for b in range(B):
    t1=rebuilds[b]; t0=t1-W
    if t1+HOLD>=T: continue
    segn='dev' if seg_mask(t1,DEV) else ('val' if seg_mask(t1,VAL) else None)
    if segn is None: continue
    nbr=nbr_resid[b]; wgt=wgt_resid[b]
    valid=nbr[:,0]>=0; idx=np.where(valid)[0]
    nb=nbr[idx]; wv=wgt[idx]
    base=logc[t0-1] if t0>0 else np.zeros(N)
    Pw=np.exp(logc[t0:t1]-base)           # (W,N) 归一化价格
    Pt=Pw[-1]
    si_p=Pw[:,idx]; nbc=np.where(nb>=0,nb,0)
    sj_p=Pw[:,nbc.ravel()].reshape(W,len(idx),-1)
    s_tr=si_p[:,:,None]-sj_p
    mu=s_tr.mean(0); sd=s_tr.std(0)+1e-9
    z=(s_tr[-1]-mu)/sd                     # 当前 z
    f_i=fwd[t1][idx][:,None]; f_j=fwd[t1][nbc]
    pnl=np.sign(-z)*(f_i-f_j)              # 收敛则为正
    ok=(nb>=0)&~np.isnan(pnl)&(np.abs(z)>1.0)
    cs=wv[ok]; pn=pnl[ok]
    if len(pn)<30: continue
    qs=np.quantile(cs,[1/3,2/3])
    for q,(lo,hi) in enumerate([(-9,qs[0]),(qs[0],qs[1]),(qs[1],9)]):
        m=(cs>lo)&(cs<=hi)
        if m.sum()>5: pair_out[segn][q].append(float(pn[m].mean()))
pair_stats={}
for segn in ['dev','val']:
    pair_stats[segn]={}
    for q in [0,1,2]:
        a=np.array(pair_out[segn][q])
        pair_stats[segn][f'corr_q{q}']={'mean_bp':round(float(a.mean()*1e4),1),
            't':round(float(a.mean()/(a.std()+1e-12)*np.sqrt(len(a))),2),'n':len(a)}
print('pair convergence by corr tercile:',json.dumps(pair_stats),flush=True)
res['pair_convergence']=pair_stats
with open(f'{OUT}/metrics_v3_anchor.json','w') as fh: json.dump(res,fh,ensure_ascii=False,indent=1)
print('done')
