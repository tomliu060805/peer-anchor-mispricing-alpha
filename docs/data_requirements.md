# 数据依赖说明

本项目依赖内部行情库。外部复现需按下表准备数据并通过环境变量指向(见 `.env.example`)。

## 必需

| 变量 | 内容 | 用途 | 频率/跨度 |
|---|---|---|---|
| `STOCK_ROOT` | 日频 OHLC + paused/high_limit/low_limit + pre_close | 收益、涨跌停、活跃域 | 日频 2014- |
| | `info/st_info` | ST 剔除 | 日频 |
| | `info/industry`(申万一级) | 行业残差化 | 月度快照 |
| | `fundamental/financial_indicator`(roe, pub_date, stat_date) | ROE gap / ΔROE | 逐日快照 |
| `INDEX_ROOT` | `price_daily`(000985 等)、`price_1m`(CNI2000/CSI1000/CSI500) | 基准、beta 引擎 | 日频/1分钟 |
| `TICK_ROOT` | `SEL2_TRANSACTION`(沪) / `SZL2_TRADE`(深) | 逐笔行为特征 | 逐笔 2015-12- |

## 可选

| 变量 | 用途 | 缺失影响 |
|---|---|---|
| `BARRA_ROOT` | 风格中性化验真 | 仅影响 `35_vs_plain.py` 等验证脚本 |
| `MONEYFLOW_ROOT` | 日频大单资金流 | 可由 `43_mf_backfill.py` 从逐笔自建 |
| `EXT_*` | 与外部项目对照的诊断脚本 | 主线不受影响 |

## 关键数据口径

- **收益**: `close/pre_close - 1`,`pre_close` 为复权口径昨收,自动处理除权除息
- **逐笔主动方向**: 沪市用 `BuySellFlag`;深市用 `BuyOrderID > SellOrderID` 判定主买
- **深市撤单**: `TradeType=='4'` 必须 null-safe 剔除(`~(x=='4').fill_null(False)`)
- **财务 PIT**: 仅使用 `pub_date <= t-1` 的记录;ΔROE 仅在 `stat_date` 变化时计算
- **已知事故**: L2 数据 2023-05-31~06-30 曾有零填充,已用 TAQ 重拉修复
