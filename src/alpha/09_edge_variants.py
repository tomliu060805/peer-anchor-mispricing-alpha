# -*- coding: utf-8 -*-
"""v7: 连边方式变体 + 因子优化, 全市场, 统一锚质量评估.
网络: price(基线K5) / volume(异常量能残差corr) / dual(价量都强) / limitco(涨停共现) / leadlag(±2最大正相关)
优化: mom10 vs mom20; gap/自身波动 标准化."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, pickle, numpy as np

CACHE=PROJ+'/cache'
OUT=PROJ+'/output/alpha'
W,REBUILD,K,MOM,HOLD,COST=120,21,5,20,5,0.00125
WLIM=250
DEV=('2015-01-01','2022-12-31'); VAL=('2023-01-01','2024-08-16')

g=np.load(f'{CACHE}/daily_grid.npz')
ret,dates,codes=g['ret'],g['dates'],g['codes']; T,N=ret.shape
money=g['money']; paused,at_hl,at_ll=g['paused'],g['at_hlimit'],g['at_llimit']
st=np.load(f'{CACHE}/st_grid.npz')['is_st']
mcap=np.load(f'{CACHE}/mcap_grid.npz')['mcap']
with open(f'{CACHE}/industry_monthly.pkl','rb') as fh: ind_map=pickle.load(fh)
ind_months=sorted(ind_map.keys())
ret0=np.nan_to_num(ret,nan=0.0); logc=np.cumsum(np.log1p(ret0),0)
mom20=np.full((T,N),np.nan,np.float32); mom20[20:]=(logc[20:]-logc[:-20]).astype(np.float32)
mom10=np.full((T,N),np.nan,np.float32); mom10[10:]=(logc[10:]-logc[:-10]).astype(np.float32)
# 20日波动
vol20=np.full((T,N),np.nan,np.float32)
r2=np.nan_to_num(ret,nan=0.0)**2; c2=np.cumsum(r2,0)
vol20[20:]=np.sqrt((c2[20:]-c2[:-20])/20).astype(np.float32)
tradable=(paused<0.5)&(at_hl<0.5)&(at_ll<0.5)&(st<0.5)&~np.isnan(ret)
dnum=dates.astype('datetime64[D]')
# 异常量能: log(money), 后续窗口内自身demean+截面市场因子剔除
lm=np.log(np.where(money>0,money,np.nan))
fwd=np.full((T,N),np.nan,np.float32)
fwd[:T-HOLD]=(np.exp(logc[HOLD:]-logc[:-HOLD])-1).astype(np.float32)

rebuilds=list(range(max(W,WLIM),T,REBUILD)); B=len(rebuilds)
def industry_of(t1):
    dstr=str(dates[t1-1]); mk=[m for m in ind_months if m<=dstr]
    imap=ind_map[mk[-1]] if mk else {}
    return np.array([imap.get(c,'') for c in codes])

def topk_from_corr(C,idx,k=K):
    n=C.shape[0]; np.fill_diagonal(C,-9)
    part=np.argpartition(-C,k,axis=1)[:,:k]
    rows=np.arange(n)[:,None]; vals=C[rows,part]
    order=np.argsort(-vals,axis=1)
    return idx[part[rows,order]], vals[rows,order]

def build_nets():
    nets={nm:{} for nm in ['price','volume','dual','limitco','leadlag']}
    for b,t1 in enumerate(rebuilds):
        Rw=ret[t1-W:t1]
        valid=(~np.isnan(Rw)).sum(0)>=110
        valid&=~(st[t1-1]==1)
        inds=industry_of(t1); valid&=inds!=''
        idx=np.where(valid)[0]; n=len(idx)
        X=np.nan_to_num(Rw[:,idx],nan=0.0).astype(np.float64)
        mkt=X.mean(1,keepdims=True)
        beta=(X*mkt).sum(0)/np.maximum((mkt*mkt).sum(),1e-12)
        Xr=X-mkt@beta[None,:]
        gi=inds[idx]
        for u in np.unique(gi):
            sel=gi==u
            if sel.sum()>1: Xr[:,sel]-=Xr[:,sel].mean(1,keepdims=True)
        sd=Xr.std(0)+1e-12; Zp=(Xr-Xr.mean(0))/sd
        Cp=(Zp.T@Zp/W).astype(np.float32)
        nets['price'][t1]=(idx,)+topk_from_corr(Cp.copy(),idx)
        # volume
        V=lm[t1-W:t1][:,idx]
        V=np.where(np.isnan(V),np.nanmean(V,0,keepdims=True),V)
        V=np.nan_to_num(V,nan=0.0)
        V=V-V.mean(0)                      # 每股demean
        vm=V.mean(1,keepdims=True)         # 市场量能因子
        bv=(V*vm).sum(0)/np.maximum((vm*vm).sum(),1e-12)
        Vr=V-vm@bv[None,:]
        for u in np.unique(gi):
            sel=gi==u
            if sel.sum()>1: Vr[:,sel]-=Vr[:,sel].mean(1,keepdims=True)
        sdv=Vr.std(0)+1e-12; Zv=(Vr-Vr.mean(0))/sdv
        Cv=(Zv.T@Zv/W).astype(np.float32)
        nets['volume'][t1]=(idx,)+topk_from_corr(Cv.copy(),idx)
        # dual: 价量秩和 (corr各自截面转秩后求和), 要求两边都靠前
        rp=np.argsort(np.argsort(-Cp,axis=1),axis=1)   # 0=最强
        rv=np.argsort(np.argsort(-Cv,axis=1),axis=1)
        Cd=-(rp+rv).astype(np.float32)
        np.fill_diagonal(Cd,-1e9)
        nb_d,wv_d=topk_from_corr(Cd,idx)
        # dual 权重用价格corr
        rows=np.arange(n)[:,None]
        loc=np.searchsorted(idx,nb_d)
        wv_price=Cp[rows,loc]
        nets['dual'][t1]=(idx,nb_d,wv_price)
        # limitco: 250日涨停共现
        A=np.nan_to_num(at_hl[t1-WLIM:t1][:,idx],nan=0.0).astype(np.float32)
        Cl=(A.T@A)
        np.fill_diagonal(Cl,-9)
        nb_l,wv_l=topk_from_corr(Cl,idx)
        wv_l=np.where(wv_l>=2,wv_l,0)      # 至少2次共现才算边
        nets['limitco'][t1]=(idx,nb_l,wv_l)
        # leadlag: lag∈{-2..2} 最大正相关
        Cll=Cp.copy()
        for L in [1,2]:
            Za=Zp[:-L]; Zb=Zp[L:]
            C1=(Za.T@Zb/ (W-L)).astype(np.float32)   # i领先j
            Cll=np.maximum(Cll,np.maximum(C1,C1.T))
        np.fill_diagonal(Cll,-9)
        nets['leadlag'][t1]=(idx,)+topk_from_corr(Cll,idx)
        if b%20==0: print(f'rebuild {b}/{B} {dates[t1-1]}',flush=True)
    return nets
nets=build_nets()
rb=np.array(rebuilds)
sig_days=np.arange(rebuilds[0]+1,T-1,HOLD)

def peer_gap(netname,momM=mom20,volnorm=False):
    S=np.full((len(sig_days),N),np.nan,np.float32)
    for si,t in enumerate(sig_days):
        t1=rb[np.searchsorted(rb,t,side='right')-1]
        idx,nb,wv=nets[netname][t1]
        m=momM[t]
        pm=m[nb]; w=np.maximum(wv,0)
        msk=~np.isnan(pm); w=w*msk
        sw=w.sum(1); ok=sw>1e-9
        agg=(np.nan_to_num(pm)*w).sum(1)/np.maximum(sw,1e-9)
        gap=agg-m[idx]
        if volnorm: gap=gap/np.maximum(vol20[t,idx]*np.sqrt(20),1e-4)
        S[si,idx[ok]]=gap[ok]
    return S

def seg_mask(t,a,b): return (dnum[t]>=np.datetime64(a))and(dnum[t]<=np.datetime64(b))
def evaluate(S,name):
    res={}; ann=252/HOLD
    for segn,(a,b) in [('dev',DEV),('val',VAL)]:
        ics=[];ls=[];te=[];prev=None;cov=[]
        for si,t in enumerate(sig_days):
            if not seg_mask(t,a,b): continue
            s=S[si].copy(); s[~tradable[t]]=np.nan; f=fwd[t]
            m=~np.isnan(s)&~np.isnan(f)
            cov.append(int(m.sum()))
            if m.sum()<100: prev=None; continue
            sv,fv=s[m],f[m]; ids=np.where(m)[0]
            xr=np.argsort(np.argsort(sv)).astype(float); yr=np.argsort(np.argsort(fv)).astype(float)
            xr=(xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
            ics.append(float((xr*yr).mean()))
            q=np.argsort(np.argsort(sv))/len(sv)
            top=set(ids[q>=0.9])
            to=1-len(top&prev)/max(len(top),1) if prev else 1.0
            rt=fv[q>=0.9].mean(); rb2=fv[q<0.1].mean(); ru=fv.mean()
            ls.append(rt-rb2-to*2*COST*2); te.append(rt-ru-to*2*COST)
            prev=top
        ics,lsv,tev=map(np.array,(ics,ls,te))
        res[segn]={'IC':round(float(ics.mean()),4),'ICIR':round(float(ics.mean()/(ics.std()+1e-12)),3),
          'Sharpe':round(float(lsv.mean()/(lsv.std()+1e-12)*np.sqrt(ann)),2),
          'TopEx_ann':round(float(tev.mean()*ann),3),'cov':int(np.mean(cov))}
    print(name,json.dumps(res),flush=True)
    return res

res={}
for nm in ['price','volume','dual','limitco','leadlag']:
    res[f'{nm}:gap20']=evaluate(peer_gap(nm),f'{nm}:gap20')
res['price:gap10']=evaluate(peer_gap('price',momM=mom10),'price:gap10')
res['price:gap20_volnorm']=evaluate(peer_gap('price',volnorm=True),'price:gap20_volnorm')
res['dual:gap20_volnorm']=evaluate(peer_gap('dual',volnorm=True),'dual:gap20_volnorm')
json.dump(res,open(f'{OUT}/metrics_v7_edges.json','w'),ensure_ascii=False,indent=1)
print('done')
