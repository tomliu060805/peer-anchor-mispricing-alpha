# -*- coding: utf-8 -*-
"""生产流水线：从原始数据到当期持仓建议。

设计原则
--------
1. **只建 v28 需要的东西**。研究期的缓存散落在 90+ 个脚本里作为副产品
   (nets_v8 藏在涨停共现探测、quanzhi_ret 藏在条件化实验)，生产不能依赖这些。
2. **串行依赖，显式声明**。研究期曾因并发导致逐笔脚本读到重建前的日期表而白跑一轮。
3. **增量优先**。逐笔特征按日增量，网格与网络按需重建。
4. **股票数变化即全量重建**。新股上市会让 codes 维度变化，所有按索引对齐的缓存
   必须同步重建，否则索引错位——这类 bug 不报错但结果全错。

阶段
----
  S1 daily      日线网格 / ST / 市值 / 行业        <- STOCK_ROOT
  S2 tick       逐笔行为特征(增量) -> tb_grid      <- TICK_ROOT
  S3 fundamental 财务网格 (ROE / dROE, PIT)         <- STOCK_ROOT
  S4 intraday   30分钟网格 + 基准收益              <- STOCK_ROOT / INDEX_ROOT
  S5 networks   价格K5 / 价量双确认 / 转移熵        <- 依赖 S1,S2
  S6 portfolio  当期打分与持仓建议                  <- 依赖全部

用法
----
  python src/live/pipeline.py --stage all        # 全量
  python src/live/pipeline.py --stage portfolio  # 仅重算组合(缓存已就绪)
  python src/live/pipeline.py --check            # 只查数据新鲜度与缓存一致性
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import (PROJ_ROOT as PROJ, CACHE_DIR as CACHE, OUTPUT_DIR,
                   STOCK_ROOT, INDEX_ROOT, TICK_ROOT)

import os, sys, json, time, pickle, argparse, subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

W, K, MOM, WLIM = 120, 5, 20, 250
REBUILD_EVERY = 21
N_WORKERS = int(os.environ.get('N_WORKERS', '48'))
BENCH_CODE = '000985.XSHG'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ============================== S1 日线 ==============================
def stage_daily():
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(f for f in os.listdir(f'{STOCK_ROOT}/price/price_daily') if f.endswith('.parquet'))
    log(f'S1 日线: {len(files)} 个交易日')
    with ProcessPoolExecutor(N_WORKERS) as ex:
        parts = list(ex.map(_read_daily, files, chunksize=16))
    df = pd.concat(parts, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df['ret'] = df['close'] / df['pre_close'] - 1.0
    df['at_hlimit'] = (df['close'] >= df['high_limit'] - 1e-9)
    df['at_llimit'] = (df['close'] <= df['low_limit'] + 1e-9)
    piv = lambda c: df.pivot_table(index='date', columns='code', values=c, aggfunc='first').astype('float32')
    ret = piv('ret')
    dates = ret.index.strftime('%Y-%m-%d').values.astype('U10')
    codes = ret.columns.values.astype('U11')
    np.savez_compressed(f'{CACHE}/daily_grid.npz', dates=dates, codes=codes,
                        ret=ret.values, close=piv('close').values, open=piv('open').values,
                        money=piv('money').values, paused=piv('paused').values,
                        at_hlimit=piv('at_hlimit').values, at_llimit=piv('at_llimit').values)
    log(f'  daily_grid {ret.shape} -> {dates[-1]}')

    stf = sorted(os.listdir(f'{STOCK_ROOT}/info/st_info'))
    with ProcessPoolExecutor(N_WORKERS) as ex:
        sts = list(ex.map(pd.read_parquet, [f'{STOCK_ROOT}/info/st_info/{f}' for f in stf], chunksize=32))
    st = pd.concat(sts, ignore_index=True)
    st['date'] = pd.to_datetime(st['date'])
    stp = st.pivot_table(index='date', columns='code', values='is_st', aggfunc='first')
    stp = stp.astype('float32').reindex(index=ret.index, columns=ret.columns)
    np.savez_compressed(f'{CACHE}/st_grid.npz', dates=dates, codes=codes, is_st=stp.values)

    # 行业(月度快照, PIT)
    inf = sorted(os.listdir(f'{STOCK_ROOT}/info/industry'))
    mfirst = ret.index.to_series().groupby(ret.index.to_period('M')).first()
    keep = {d.strftime('%Y-%m-%d') + '.parquet' for d in mfirst}
    ind_map = {}
    for f in [x for x in inf if x in keep]:
        d = pd.read_parquet(f'{STOCK_ROOT}/info/industry/{f}')
        ds = d[d['category'] == 'sw_l1']
        if len(ds) == 0:
            ds = d[d['category'] == 'jq_l1']
        ind_map[f[:10]] = dict(zip(ds['code'], ds['industry_name']))
    with open(f'{CACHE}/industry_monthly.pkl', 'wb') as fh:
        pickle.dump(ind_map, fh)
    log(f'  st_grid / industry({len(ind_map)}月) 完成')
    return dates, codes


# ============================== S2 逐笔 ==============================
_PREF = ("00", "30", "60", "68")


def _tick_one(ds):
    import polars as pl
    out = f'{CACHE}/tickbehav/{ds}.parquet'
    if os.path.exists(out):
        return ds, 'skip'
    try:
        parts = []
        for sub, is_sh in (('SEL2_TRANSACTION', True), ('SZL2_TRADE', False)):
            p = f'{TICK_ROOT}/{sub}/{ds}.parquet'
            if not os.path.exists(p):
                continue
            if is_sh:
                lf = (pl.scan_parquet(p)
                      .select(pl.col('Symbol').alias('code'),
                              pl.col('TradeAmount').fill_null(0.0).alias('amount'),
                              pl.col('BuySellFlag').alias('flag'),
                              pl.col('BuyRecID'), pl.col('SellRecID'))
                      .filter(pl.col('code').str.slice(0, 2).is_in(list(_PREF))))
                sell = lf.filter(pl.col('flag') == 'S').rename({'SellRecID': 'oid'})
                buy = lf.filter(pl.col('flag') == 'B').rename({'BuyRecID': 'oid'})
                ex = 'SH'
            else:
                lf = (pl.scan_parquet(p)
                      .filter(~(pl.col('TradeType') == '4').fill_null(False))
                      .select(pl.col('Symbol').alias('code'),
                              (pl.col('TradePrice') * pl.col('TradeVolume')).fill_null(0.0).alias('amount'),
                              pl.col('BuyOrderID'), pl.col('SellOrderID'))
                      .filter(pl.col('code').str.slice(0, 2).is_in(list(_PREF))))
                sell = lf.filter(pl.col('SellOrderID') > pl.col('BuyOrderID')).rename({'SellOrderID': 'oid'})
                buy = lf.filter(pl.col('BuyOrderID') > pl.col('SellOrderID')).rename({'BuyOrderID': 'oid'})
                ex = 'SZ'
            tm = lf.group_by('code').agg(pl.col('amount').sum().alias('total_money'))

            def agg(side, tag):
                per = side.group_by(['code', 'oid']).agg(pl.col('amount').sum().alias('amt'),
                                                         pl.len().alias('pieces'))
                return per.group_by('code').agg(
                    pl.col('amt').sum().alias(f'{tag}_amt'),
                    pl.col('amt').filter(pl.col('pieces') >= 3).sum().alias(f'{tag}_sweep'),
                    pl.len().cast(pl.Int64).alias(f'{tag}_orders'))
            d = (tm.join(agg(sell, 'asell'), on='code', how='left')
                   .join(agg(buy, 'abuy'), on='code', how='left')
                   .with_columns(pl.lit(ex).alias('exchange')))
            parts.append(d.collect(engine='streaming'))
        if not parts:
            return ds, 'nofile'
        import polars as pl2
        pl2.concat(parts, how='vertical').with_columns(pl2.lit(ds).alias('date')) \
            .write_parquet(out + '.tmp')
        os.rename(out + '.tmp', out)
        return ds, 'ok'
    except Exception as e:
        return ds, f'ERR:{type(e).__name__}:{e}'


_CTX = {}
_TB_COLS = ['total_money', 'asell_amt', 'asell_sweep', 'asell_orders', 'abuy_amt', 'abuy_sweep']


def _init_ctx(dates, codes):
    _CTX['dates'] = dates
    _CTX['codes'] = codes
    _CTX['cix'] = {c: i for i, c in enumerate(codes)}
    _CTX['N'] = len(codes)


def _load_tb(i):
    dates, cix, Nn = _CTX['dates'], _CTX['cix'], _CTX['N']
    ds = str(dates[i])
    f = f'{CACHE}/tickbehav/{ds}.parquet'
    out = np.full((len(_TB_COLS), Nn), np.nan, np.float32)
    if not os.path.exists(f):
        return i, out
    d = pd.read_parquet(f)
    suf = np.where(d['exchange'].values == 'SZ', '.XSHE', '.XSHG')
    full = np.char.add(d['code'].values.astype(str), suf)
    ix = np.array([cix.get(c, -1) for c in full])
    ok = ix >= 0
    for j, c in enumerate(_TB_COLS):
        v = d[c].values.astype(np.float32) if c in d.columns else np.zeros(len(d), np.float32)
        out[j, ix[ok]] = np.nan_to_num(v[ok], nan=0.0)
    return i, out


_FCOLS = ['roe', 'inc_net_profit_year_on_year', 'gross_profit_margin', 'net_profit_margin']


def _load_fi(i):
    """财务网格 6 列: roe / SUE / 毛利率 / 净利率 / stat_date序号 / 新鲜度。

    v28 打分只用 roe(第0列)与 stat_date(第4列, 用于识别新报告算 dROE),
    其余三列供研究链的变体实验使用——列序必须与研究期一致, 否则解包错位。
    PIT: pub_date <= t-1 日。
    """
    dates, cix, Nn = _CTX['dates'], _CTX['cix'], _CTX['N']
    root = f'{STOCK_ROOT}/fundamental/financial_indicator'
    ds = str(dates[i])
    out = np.full((len(_FCOLS) + 2, Nn), np.nan, np.float32)
    f = f'{root}/{ds}.parquet'
    if not os.path.exists(f):
        return i, out
    try:
        d = pd.read_parquet(f, columns=['code', 'pub_date', 'stat_date'] + _FCOLS)
    except Exception:
        return i, out
    pdte = pd.to_datetime(d['pub_date'], errors='coerce')
    d = d[pdte <= pd.Timestamp(ds) - pd.Timedelta(days=1)]
    if len(d) == 0:
        return i, out
    ix = np.array([cix.get(c, -1) for c in d['code']])
    ok = ix >= 0
    for j, c in enumerate(_FCOLS):
        v = pd.to_numeric(d[c], errors='coerce').values.astype(np.float32)
        out[j, ix[ok]] = v[ok]
    sd = pd.to_datetime(d['stat_date'], errors='coerce')
    out[len(_FCOLS), ix[ok]] = (sd.values.astype('datetime64[D]').astype(np.float32))[ok]
    pdv = pd.to_datetime(d['pub_date'], errors='coerce')
    out[len(_FCOLS) + 1, ix[ok]] = (pd.Timestamp(ds) - pdv).dt.days.values.astype(np.float32)[ok]
    return i, out


def _load_c30(i):
    dates, cix, Nn = _CTX['dates'], _CTX['cix'], _CTX['N']
    ds = str(dates[i])
    f = f'{STOCK_ROOT}/price/price_30m/{ds}.parquet'
    out = np.full((8, Nn), np.nan, np.float32)
    if not os.path.exists(f):
        return i, out
    d = pd.read_parquet(f, columns=['datetime', 'code', 'close'])
    bars = sorted(d['datetime'].unique())[:8]
    bm = {b: j for j, b in enumerate(bars)}
    ix = np.array([cix.get(c, -1) for c in d['code']])
    bj = np.array([bm.get(b, -1) for b in d['datetime']])
    v = d['close'].values.astype(np.float32)
    ok = (ix >= 0) & (bj >= 0)
    out[bj[ok], ix[ok]] = v[ok]
    return i, out


def _load_bench(i):
    dates = _CTX['dates']
    f = f'{INDEX_ROOT}/price/price_daily/{dates[i]}.parquet'
    if not os.path.exists(f):
        return i, np.nan
    d = pd.read_parquet(f, columns=['code', 'close', 'pre_close'])
    r = d[d['code'] == BENCH_CODE]
    if len(r) == 0:
        return i, np.nan
    return i, float(r['close'].iloc[0]) / float(r['pre_close'].iloc[0]) - 1


def _read_daily(f):
    return pd.read_parquet(f'{STOCK_ROOT}/price/price_daily/{f}',
                           columns=['date', 'code', 'close', 'pre_close', 'open',
                                    'money', 'paused', 'high_limit', 'low_limit'])


def stage_tick(dates, codes):
    os.makedirs(f'{CACHE}/tickbehav', exist_ok=True)
    need = [str(d) for d in dates if str(d) >= '2015-12-01']
    todo = [d for d in need if not os.path.exists(f'{CACHE}/tickbehav/{d}.parquet')]
    log(f'S2 逐笔: 需 {len(need)} 天, 待建 {len(todo)} 天')
    if todo:
        errs = 0
        with ProcessPoolExecutor(N_WORKERS) as ex:
            for i, (d, st) in enumerate(ex.map(_tick_one, todo, chunksize=1)):
                if st.startswith('ERR'):
                    errs += 1
                    if errs <= 3:
                        log(f'  {d} {st}')
                if i % 200 == 0:
                    log(f'  {i}/{len(todo)}')
        log(f'  完成, 错误 {errs}')
    # 汇总为网格
    Nn = len(codes)
    _init_ctx(dates, codes)
    with ProcessPoolExecutor(N_WORKERS) as ex:
        outs = list(ex.map(_load_tb, range(len(dates)), chunksize=8))
    TB = np.full((len(dates), len(_TB_COLS), Nn), np.nan, np.float32)
    for i, o in outs:
        TB[i] = o
    np.savez_compressed(f'{CACHE}/tb_grid.npz', tb=TB)
    log(f'  tb_grid {TB.shape}')


# ============================== S3 财务 ==============================
def stage_fundamental(dates, codes):
    _init_ctx(dates, codes)
    log('S3 财务网格...')
    with ProcessPoolExecutor(N_WORKERS) as ex:
        outs = list(ex.map(_load_fi, range(len(dates)), chunksize=16))
    FI = np.full((len(dates), len(_FCOLS) + 2, len(codes)), np.nan, np.float32)
    for i, o in outs:
        FI[i] = o
    np.savez_compressed(f'{CACHE}/fi_grid.npz', fi=FI)
    log(f'  fi_grid {FI.shape}')


# ============================== S4 日内与基准 ==============================
def stage_intraday(dates, codes):
    _init_ctx(dates, codes)
    log('S4 30分钟网格...')
    with ProcessPoolExecutor(N_WORKERS) as ex:
        outs = list(ex.map(_load_c30, range(len(dates)), chunksize=8))
    C30 = np.full((len(dates), 8, len(codes)), np.nan, np.float32)
    for i, o in outs:
        C30[i] = o
    np.savez_compressed(f'{CACHE}/c30_grid.npz', c=C30)
    with ProcessPoolExecutor(N_WORKERS) as ex:
        outs = list(ex.map(_load_bench, range(len(dates)), chunksize=16))
    qz = np.zeros(len(dates))
    for i, val in outs:
        if val == val:
            qz[i] = val
    np.save(f'{CACHE}/quanzhi_ret.npy', qz)
    log(f'  c30_grid {C30.shape} / benchmark 完成')


# ============================== S5 网络 ==============================
_NETIN = {}


def _netin():
    """每个 worker 进程只加载一次网络输入 (原先每任务加载, 90 worker 会把内存打爆)。"""
    if not _NETIN:
        z = np.load(f'{CACHE}/_netinput.npz')
        _NETIN['ret'] = z['ret']
        _NETIN['st'] = z['st']
        _NETIN['inds'] = z['inds']      # int16 行业编码, -1 表示缺失
        _NETIN['nf'] = z['nf']
    return _NETIN


def _build_net(args):
    t1, kind = args[0], args[1]
    Kn = args[2] if len(args) > 2 else K
    z = _netin()
    ret, st_g, inds_all = z['ret'], z['st'], z['inds']
    NF = z['nf'] if kind in ('dual',) else None
    Nn = ret.shape[1]
    Rw = ret[t1 - W:t1]
    valid = (~np.isnan(Rw)).sum(0) >= 110
    valid &= ~(st_g[t1 - 1] == 1)
    inds = inds_all[t1]
    valid &= inds >= 0
    idx = np.where(valid)[0]
    n = len(idx)
    on = np.full((Nn, Kn), -1, np.int32)
    ow = np.zeros((Nn, Kn), np.float32)
    if n < 200:
        return t1, on, ow

    def resid(M):
        X = np.nan_to_num(M[:, idx], nan=0.0).astype(np.float64)
        mkt = X.mean(1, keepdims=True)
        beta = (X * mkt).sum(0) / np.maximum((mkt * mkt).sum(), 1e-12)
        Xr = X - mkt @ beta[None, :]
        gi = inds[idx]
        for u in np.unique(gi):
            sel = gi == u
            if sel.sum() > 1:
                Xr[:, sel] -= Xr[:, sel].mean(1, keepdims=True)
        sd = Xr.std(0) + 1e-12
        return (Xr - Xr.mean(0)) / sd

    Zp = resid(Rw)
    Cp = (Zp.T @ Zp / W).astype(np.float32)
    np.fill_diagonal(Cp, -9)
    if kind == 'price':
        C = Cp
    elif kind == 'dual':
        Zf = resid(NF[t1 - W:t1])
        Cf = (Zf.T @ Zf / W).astype(np.float32)
        np.fill_diagonal(Cf, -9)
        rp = np.argsort(np.argsort(-Cp, axis=1), axis=1)
        rf = np.argsort(np.argsort(-Cf, axis=1), axis=1)
        C = np.where((rp < 30) & (rf < 30), Cp, -9).astype(np.float32)
    else:                                   # te 转移熵
        cand = np.argpartition(-Cp, 30, axis=1)[:, :30]
        S = np.sign(Zp)
        S = np.where(np.abs(Zp) < 0.5, 0, S).astype(np.int8) + 1     # {0,1,2}
        C = np.full((n, n), -9, np.float32)
        for i in range(n):
            xi = S[1:, i]
            xi_1 = S[:-1, i]
            for j in cand[i]:
                if j == i:
                    continue
                yj = S[:-1, j]
                cnt = np.zeros((3, 3, 3))
                np.add.at(cnt, (xi, xi_1, yj), 1)
                p = cnt / cnt.sum()
                p_xy = p.sum(0)
                p_x1 = p.sum((0, 2))
                p_xx = p.sum(2)
                te = 0.0
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            if p[a, b, c] > 0 and p_xy[b, c] > 0 and p_x1[b] > 0 and p_xx[a, b] > 0:
                                te += p[a, b, c] * np.log(p[a, b, c] * p_x1[b] / (p_xy[b, c] * p_xx[a, b]))
                C[i, j] = te
    part = np.argpartition(-C, Kn, axis=1)[:, :Kn]
    rows = np.arange(n)[:, None]
    vals = C[rows, part]
    o = np.argsort(-vals, axis=1)
    nb, wv = part[rows, o], vals[rows, o]
    ok = wv > -8
    on[idx] = np.where(ok, idx[nb], -1)
    ow[idx] = np.maximum(np.where(ok, wv, 0), 0)
    return t1, on, ow


def _prep_net_input(dates, codes):
    """写 _netinput.npz 供 worker 读。行业存 int16 而非 U12 字符串——
    后者是 3079x5469x48B ~ 808MB, 每个 worker 载一份会撑爆内存。"""
    g = np.load(f'{CACHE}/daily_grid.npz')
    ret = g['ret']
    st_g = np.load(f'{CACHE}/st_grid.npz')['is_st']
    Tn, Nn = ret.shape
    with open(f'{CACHE}/industry_monthly.pkl', 'rb') as fh:
        im = pickle.load(fh)
    months = sorted(im.keys())
    vocab = {}
    for m in months:
        for u in im[m].values():
            vocab.setdefault(u, len(vocab))
    inds_all = np.full((Tn, Nn), -1, np.int16)
    cur = np.full(Nn, -1, np.int16)
    mi = 0
    for t in range(Tn):
        d = str(dates[t - 1]) if t > 0 else str(dates[0])
        while mi + 1 < len(months) and months[mi + 1] <= d:
            mi += 1
        if months and months[mi] <= d:
            mp = im[months[mi]]
            cur = np.array([vocab.get(mp.get(c, ''), -1) for c in codes], np.int16)
        inds_all[t] = cur
    TB = np.load(f'{CACHE}/tb_grid.npz')['tb']
    with np.errstate(all='ignore'):
        NF = (TB[:, 4] - TB[:, 1]) / np.maximum(TB[:, 0], 1.0)
    NF = np.where(np.isfinite(NF) & (TB[:, 0] > 0), NF, np.nan).astype(np.float32)
    np.savez(f'{CACHE}/_netinput.npz', ret=ret, st=st_g, inds=inds_all, nf=NF)
    return Tn, Nn


def stage_networks(dates, codes):
    Tn, Nn = _prep_net_input(dates, codes)
    rebuilds = list(range(WLIM, Tn, REBUILD_EVERY))
    log(f'S5 网络: {len(rebuilds)} 个重建点 x 3 类')
    for kind, fn in [('price', 'nets_price'), ('dual', 'nets_dual'), ('te', 'nets_te')]:
        t0 = time.time()
        with ProcessPoolExecutor(N_WORKERS if kind != 'te' else max(N_WORKERS // 2, 8)) as ex:
            outs = list(ex.map(_build_net, [(t, kind) for t in rebuilds], chunksize=1))
        outs.sort(key=lambda x: x[0])
        np.savez_compressed(f'{CACHE}/{fn}.npz',
                            t=np.array([o[0] for o in outs], np.int32),
                            n=np.stack([o[1] for o in outs]),
                            w=np.stack([o[2] for o in outs]))
        log(f'  {kind} 完成 ({time.time()-t0:.0f}s)')
    os.remove(f'{CACHE}/_netinput.npz')


# ============================== 检查 ==============================
def stage_check():
    g = np.load(f'{CACHE}/daily_grid.npz')
    dates, codes = g['dates'], g['codes']
    Tn, Nn = g['ret'].shape
    src_last = sorted(os.listdir(f'{STOCK_ROOT}/price/price_daily'))[-1][:10]
    print(f'源数据最新: {src_last} | 缓存最新: {dates[-1]} | 维度 ({Tn}, {Nn})')
    ok = True
    for f, key in [('st_grid', 'is_st'), ('tb_grid', 'tb'), ('fi_grid', 'fi'), ('c30_grid', 'c')]:
        p = f'{CACHE}/{f}.npz'
        if not os.path.exists(p):
            print(f'  ✗ {f} 缺失')
            ok = False
            continue
        a = np.load(p)[key]
        good = (a.shape[0] == Tn) and (a.shape[-1] == Nn)
        print(f'  {"✓" if good else "✗"} {f} {a.shape}')
        ok &= good
    for f in ['nets_price', 'nets_dual', 'nets_te']:
        p = f'{CACHE}/{f}.npz'
        if not os.path.exists(p):
            print(f'  ✗ {f} 缺失')
            ok = False
            continue
        z = np.load(p)
        good = z['n'].shape[1] == Nn
        print(f'  {"✓" if good else "✗"} {f} 重建点{len(z["t"])} 末点{dates[z["t"][-1]]}')
        ok &= good
    nb = len(os.listdir(f'{CACHE}/tickbehav')) if os.path.exists(f'{CACHE}/tickbehav') else 0
    print(f'  逐笔文件 {nb} 天')
    print('一致性:', '通过' if ok else '不通过(需重建)')
    return ok


# ======================== S5b 研究链兼容缓存 ========================
def stage_research_compat(dates, codes):
    """reconcile.py / phase_robustness.py 要跑研究脚本链, 而研究链读的是研究期的
    缓存名。这些是别名与补建, 不进 v28 打分, 但**必须与生产网络同维度**——
    codes 维度变化后不同步重建, 研究链会静默按错位索引出结果。"""
    log('S5b 研究链兼容缓存...')
    Nn = len(codes)
    p_ = np.load(f'{CACHE}/nets_price.npz')
    d_ = np.load(f'{CACHE}/nets_dual.npz')
    t_ = np.load(f'{CACHE}/nets_te.npz')
    assert p_['n'].shape[1] == Nn, f"nets_price 第二维 {p_['n'].shape[1]} != codes {Nn}"
    np.savez_compressed(f'{CACHE}/nets_v8.npz', rebuilds=p_['t'], nb_p=p_['n'], wv_p=p_['w'])
    np.savez_compressed(f'{CACHE}/nets_v41_flow.npz',
                        dual_t=d_['t'], dual_n=d_['n'], dual_w=d_['w'])
    np.savez_compressed(f'{CACHE}/nets_v47_te.npz', t=t_['t'], n=t_['n'], w=t_['w'])
    log('  三个别名完成')

    # nets_v17: K=20(21日重建) + K=5(10日重建), 供研究链变体实验与 Leiden 社区
    Tn, _ = _prep_net_input(dates, codes)
    rb21 = list(range(WLIM, Tn, 21))
    rb10 = list(range(WLIM, Tn, 10))
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        o21_5 = sorted(ex.map(_build_net, [(t, 'price', 5) for t in rb21], chunksize=1))
        o21_20 = sorted(ex.map(_build_net, [(t, 'price', 20) for t in rb21], chunksize=1))
        o10_5 = sorted(ex.map(_build_net, [(t, 'price', 5) for t in rb10], chunksize=1))
    np.savez_compressed(f'{CACHE}/nets_v17.npz',
                        rb21=np.array(rb21, np.int32),
                        n21_5=np.stack([o[1] for o in o21_5]),
                        w21_5=np.stack([o[2] for o in o21_5]),
                        n21_20=np.stack([o[1] for o in o21_20]),
                        w21_20=np.stack([o[2] for o in o21_20]),
                        rb10=np.array(rb10, np.int32),
                        n10_5=np.stack([o[1] for o in o10_5]),
                        w10_5=np.stack([o[2] for o in o10_5]))
    if os.path.exists(f'{CACHE}/_netinput.npz'):
        os.remove(f'{CACHE}/_netinput.npz')
    log(f'  nets_v17 完成 ({time.time() - t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'daily', 'tick', 'fundamental', 'intraday',
                             'networks', 'research', 'portfolio'])
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if stage_check() else 1)
    os.makedirs(f'{OUTPUT_DIR}/live', exist_ok=True)
    if a.stage in ('all', 'daily'):
        stage_daily()
    g = np.load(f'{CACHE}/daily_grid.npz')
    dates, codes = g['dates'], g['codes']
    if a.stage in ('all', 'tick'):
        stage_tick(dates, codes)
    if a.stage in ('all', 'fundamental'):
        stage_fundamental(dates, codes)
    if a.stage in ('all', 'intraday'):
        stage_intraday(dates, codes)
    if a.stage in ('all', 'networks'):
        stage_networks(dates, codes)
    if a.stage == 'research':
        stage_research_compat(dates, codes)
    if a.stage in ('all', 'portfolio'):
        from generate_portfolio import generate
        generate()
    log('pipeline 完成')


if __name__ == '__main__':
    main()
