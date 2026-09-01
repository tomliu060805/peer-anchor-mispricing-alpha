# 生产流水线

研究期的缓存散落在 90+ 个脚本里作为副产品——网络藏在涨停共现探测脚本、基准收益藏在
条件化实验脚本。生产不能依赖"跑某个研究脚本的副作用"，本文档描述重写后的独立流水线。

## 依赖链

```
S1 daily        日线网格 / ST / 行业月度快照        <- STOCK_ROOT
      |
S2 tick         逐笔行为特征(增量) -> tb_grid        <- TICK_ROOT
      |
S3 fundamental  ROE / stat_date (PIT: pub_date < t)  <- STOCK_ROOT
      |
S4 intraday     30分钟网格 + 基准收益                <- STOCK_ROOT / INDEX_ROOT
      |
S5 networks     价格K5 / 价量双确认 / 转移熵         <- 依赖 S1,S2
      |
S6 portfolio    当期打分与持仓建议                    <- 依赖全部
```

**必须串行**。研究期曾因并发导致逐笔脚本读到重建前的日期表，白跑一轮且不报错。

## 用法

```bash
python src/live/pipeline.py --stage all         # 全量
python src/live/pipeline.py --stage networks    # 单阶段
python src/live/pipeline.py --check             # 只查新鲜度与维度一致性
python src/live/generate_portfolio.py           # 单出组合(缓存已就绪)
bash   src/live/run_decision_day.sh             # 决策日入口(判日期+跑+出单)
```

环境变量见 `.env.example`；`N_WORKERS` 控制并行度。

## 股票数变化 = 全量重建

所有缓存按 `codes` 索引对齐。新股上市使 `codes` 维度变化(如 5468 -> 5469)后，
**所有按索引对齐的缓存必须同步重建**，否则索引错位。这类 bug 不报错但结果全错。
`--check` 会比对各缓存的第二维，不一致即拒绝出单。

## 三个并行化坑

从研究脚本搬到 100 核生产时踩的，都不是算法问题：

**1. 闭包不能跨进程 pickle。**
`ProcessPoolExecutor.map` 要求函数可 pickle，局部函数直接崩
(`Can't pickle local object`)。所有 worker 提到模块级，共享数据放模块级 `_CTX`。

**2. 每任务重载大数组会撑爆内存。**
`_build_net` 原本每次调用都 `np.load(_netinput.npz)`，其中行业网格是 `U12`
字符串数组 = 3079 × 5469 × 48B ≈ 808MB。单机顺序跑只是慢；100 worker × 135 任务
就是致命的。改为进程内缓存一次 + 行业存 `int16`(34MB，省 96%)。

**3. `pgrep -f` / `pkill -f` 会匹配到发起命令自己。**
`pkill -f "pipeline.py"` 时，执行这条命令的 shell 命令行本身含有 `pipeline.py`，
于是把自己杀掉；用 `pgrep -f` 轮询等待则永远等不到空。
两种解法：模式加字符类躲开(`pipelin[e].py`)，或把要跑的东西写成独立脚本文件
再 `setsid` 启动，让等待方匹配脚本名而非命令行。

## 决策日调度

`run_decision_day.sh` 自锚定日起每 5 个交易日为一个决策日，非决策日直接退出。
建议 crontab(收盘后数据落地)：

```
30 16 * * 1-5  /path/to/src/live/run_decision_day.sh >> /path/to/output/live/cron.log 2>&1
```

脚本在两处会主动中止而**不出单**，因为在陈旧或不一致的数据上下单比不下单更糟：
- 源数据尚未更新到当日
- `--check` 缓存一致性未通过

## 产出

```
output/live/portfolio_{date}.csv   目标持仓: 代码/权重/总分/三块分项/NEW-HOLD
output/live/orders_{date}.csv      相对上期的买卖指令
output/live/last_holdings.json     上期持仓(缓冲带判断用)
output/live/log.jsonl              append-only 决策日志(含配置 SHA)
```

配置从 `config/frozen_alpha_v28.json` 读取并记录其 SHA，配置改动必须先改 json——
避免"代码里偷偷调参"后无法追溯当期用的是哪一版。

## 执行

路径 B(项目默认)：收盘后算，T+1 上午 11:30 前卖，13:30-14:30 买。
不要开盘集中下单。

## 对拍: 生产实现 vs 回测实现

`generate_portfolio.py` 是对研究代码的**重新实现**——目的是摆脱"跑某个研究脚本
的副作用"这种依赖。但重新实现本身就是风险:如果打分逻辑与回测有出入,产出的持仓
就不是被验证过的那个策略,而回测指标会继续被当成它的业绩。

`reconcile.py` 在同一批网络输入下逐日比较两条路径选出的股票集合:

```bash
python src/live/reconcile.py     # 结果写 output/live/reconcile.json
```

判据: 交集/并集 >= 0.98 且分数相关 >= 0.999。

为让研究链能吃生产网络,`cache/` 下建了三个别名文件
(`nets_v8` / `nets_v41_flow` / `nets_v47_te`),内容分别指向
`nets_price` / `nets_dual` / `nets_te`。研究链还需要 `nets_v17`
(K=20 邻居 + 10 日重建),该缓存只被 v17 期的变体实验和 Leiden 社区计算使用,
不进入 v28 打分,但因为它在 exec 前缀里,不建就跑不起来。

## 单一打分实现

`paper_ledger.py` 只做结算与记账,打分全部委托 `generate_portfolio.generate()`。
旧的 `weekly_signal.py` 自带一套打分(且停留在 v27、没有 B2 约束),已删除——
同一策略存在多套打分实现时,它们迟早会悄悄分叉,而分叉不会报错。

## 缓存是全局共享状态: 上锁

`cache/` 下的 npz 被所有脚本共享。一个进程在重建某张网格、另一个进程同时读它,
读到的就是写了一半的文件——`zipfile.BadZipFile`。生产化过程中因此白跑了两轮:
一次是研究链正在读缓存时启动了 `pipeline.py --stage all`,一次是 kill 进程时
截断了正在写的 `tj_grid.npz`。

两条应对:

1. `run_weekly.sh` / `run_decision_day.sh` 开头用 `flock` 取排他锁,
   拿不到就直接退出。定时任务与手动运行撞车时不会互相踩。
2. 中断任务不要 `kill -9`。写到一半的 npz 不会自己修复,而且**下次读它才报错**,
   报错点离真正的原因很远。真要清理,事后逐个校验:

```bash
for f in cache/*.npz; do
  python -c "import numpy as np;z=np.load('$f');[z[k].shape for k in z.files]" \
    2>/dev/null || echo "坏: $f"
done
```

研究链会把 `mf_grid` / `tj_grid` 等缓存作为副作用重建,所以删掉坏文件后
重跑一次即可,不必手工修复。

## 配置 SHA 只覆盖参数段

决策日志每期记 `config_sha`, 用于事后按配置版本归组。哈希只取参数段
(`version/domain/score/portfolio/execution/cost/benchmark/split`),
不含 `evidence` 与 `known_limits`——补一条已知局限属于文档更新, 不该让
"同 SHA = 同配置"这个不变量失效。

哈希口径本身在 2026-09-01 变更过一次(从整文件改为参数段), 此前日志中的
`1c19b0c2122c` 与之后的 `d2bf8a08da2b` 指向的是**同一份参数**, 不是配置改动。
