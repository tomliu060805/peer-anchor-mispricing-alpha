# -*- coding: utf-8 -*-
"""测试段与全样本超额曲线 + 逐年超额(alpha 与 融合h=1.0)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
R=PROJ+''
src=open(f'{R}/src/test/open_test_segment.py',encoding='utf-8').read()
exec(src.split("out = {'criteria': CRIT}")[0])
dk,dpr=run_alpha(daily=True); dd=dnum[dk]
qz=np.load(f'{CACHE}/quanzhi_ret.npy'); dbq=qz[dk]
bd=beta_daily_pnl(dd)
m=dd>=np.datetime64('2016-01-01')
dd,dpr,dbq,bd=dd[m],dpr[m],dbq[m],bd[m]
exA=(1+dpr)/(1+dbq)-1
exF=(1+dpr+bd)/(1+dbq)-1
TEST0=np.datetime64('2024-08-19')
def st(ex,per=252):
    nav=np.cumprod(1+ex)
    return dict(ExAnn=float(nav[-1]**(per/len(ex))-1),IR=float(ex.mean()/ex.std()*np.sqrt(per)),
                ExMDD=float((1-nav/np.maximum.accumulate(nav)).max()))
years=sorted(set(str(d)[:4] for d in dd))
yrA={y:float(np.prod(1+exA[[str(x)[:4]==y for x in dd]])-1) for y in years}
yrF={y:float(np.prod(1+exF[[str(x)[:4]==y for x in dd]])-1) for y in years}
info={'full_alpha':st(exA),'full_fusion':st(exF),
      'test_alpha':st(exA[dd>=TEST0]),'test_fusion':st(exF[dd>=TEST0]),
      'yearly_alpha':{k:round(v,4) for k,v in yrA.items()},
      'yearly_fusion':{k:round(v,4) for k,v in yrF.items()}}
print(json.dumps(info,ensure_ascii=False,indent=1))
json.dump(info,open(f'{R}/output/test/test_curves.json','w'),ensure_ascii=False,indent=1)

fig=plt.figure(figsize=(14,11))
gs=fig.add_gridspec(3,2,height_ratios=[3,2,2.4],hspace=0.32,wspace=0.18)
# 全样本
ax=fig.add_subplot(gs[0,:])
ax.plot(dd,np.cumprod(1+exA),lw=1.5,color='#7f7f7f',label=f"alpha单腿: 全样本{info['full_alpha']['ExAnn']:.1%}/IR{info['full_alpha']['IR']:.2f}")
ax.plot(dd,np.cumprod(1+exF),lw=1.8,color='#d62728',label=f"融合h=1.0: 全样本{info['full_fusion']['ExAnn']:.1%}/IR{info['full_fusion']['IR']:.2f}")
ax.axvline(np.datetime64('2023-01-01'),color='gray',ls=':',lw=1)
ax.axvspan(TEST0,dd[-1],color='#2ca02c',alpha=.10)
ax.text(np.datetime64('2024-10-01'),1.15,'★ TEST 段\n(从未参与调参)',fontsize=10,color='#2ca02c',weight='bold')
ax.text(np.datetime64('2023-02-01'),1.05,'val起点',fontsize=8,color='gray')
ax.set_yscale('log'); tk=[1,1.5,2,3,4,5,6,8]
ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_minor_locator(FixedLocator([]))
f2=ScalarFormatter(); f2.set_scientific(False); ax.yaxis.set_major_formatter(f2)
ax.set_ylabel('超额净值'); ax.set_title('全样本超额净值 (对中证全指, 日频, 费后双边20bp)')
ax.legend(loc='upper left'); ax.grid(alpha=.3)
# 测试段放大
mt=dd>=TEST0
ax=fig.add_subplot(gs[1,:])
ax.plot(dd[mt],np.cumprod(1+exA[mt]),lw=1.6,color='#7f7f7f',label=f"alpha单腿: {info['test_alpha']['ExAnn']:.1%}/IR{info['test_alpha']['IR']:.2f}/MDD{info['test_alpha']['ExMDD']:.1%}")
ax.plot(dd[mt],np.cumprod(1+exF[mt]),lw=2.0,color='#d62728',label=f"融合h=1.0: {info['test_fusion']['ExAnn']:.1%}/IR{info['test_fusion']['IR']:.2f}/MDD{info['test_fusion']['ExMDD']:.1%}")
ax.axhline(1,color='k',lw=.6)
ax.set_ylabel('超额净值'); ax.set_title('★ 测试段超额净值 (2024-08-19 ~ 2026-08-06, 96周, 样本外)')
ax.legend(loc='upper left'); ax.grid(alpha=.3)
# 逐年
ax=fig.add_subplot(gs[2,:])
x=np.arange(len(years)); w=0.38
ax.bar(x-w/2,[yrA[y] for y in years],w,color='#7f7f7f',label='alpha单腿')
ax.bar(x+w/2,[yrF[y] for y in years],w,color='#d62728',label='融合h=1.0')
for i,y in enumerate(years):
    ax.text(i-w/2,yrA[y]+(0.01 if yrA[y]>=0 else -0.035),f'{yrA[y]*100:.0f}',ha='center',fontsize=8,color='#404040')
    ax.text(i+w/2,yrF[y]+(0.01 if yrF[y]>=0 else -0.035),f'{yrF[y]*100:.0f}',ha='center',fontsize=8,color='#8b0000')
for i,y in enumerate(years):
    if y in ('2024','2025','2026'): ax.axvspan(i-0.5,i+0.5,color='#2ca02c',alpha=.08)
ax.axhline(0,color='k',lw=.7); ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_title('逐年超额 (%, 复利, 费后; 绿底=测试段所含年份, 2024年为混合年)')
ax.legend(); ax.grid(alpha=.3,axis='y')
CH=f'{R}/charts/test'; os.makedirs(CH,exist_ok=True)
plt.savefig(f'{CH}/test_and_full_excess.png',dpi=130,bbox_inches='tight')
print('chart saved')
