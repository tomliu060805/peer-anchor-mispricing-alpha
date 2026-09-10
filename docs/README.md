# 文档索引

## 生产策略

| 文档 | 内容 |
|---|---|
| [`alpha_v27_spec.md`](alpha_v27_spec.md) | 选股引擎完整规格 + 组件消融表 |
| [`beta_breakout_replication.md`](beta_breakout_replication.md) | 破位择时引擎的构建、检验与判负清单 |
| [`execution.md`](execution.md) | 双路径执行代价、日内时点结构、阻塞率 |
| [`data_requirements.md`](data_requirements.md) | 数据依赖与关键口径 |

## 证据与纪律

| 文档 | 内容 |
|---|---|
| [`robustness_and_weighting.md`](robustness_and_weighting.md) | ★ 随机组合零基准 / bootstrap CI / 滚动IR / 权重研究 |
| [`open_issues.md`](open_issues.md) | ★ 测试段开封结果 + 未决事项 + 观察名单 |
| [`experiment_log.md`](experiment_log.md) | 约 240 个变体的完整实验记录(含全部判负) |
| [`live_validation.md`](live_validation.md) | 生产实现与回测的逐股对拍、相位稳健性 |

## 实盘运行

| 文档 | 内容 |
|---|---|
| [`production_pipeline.md`](production_pipeline.md) | 流水线依赖链、并行化坑、缓存加锁 |
| [`scheduling.md`](scheduling.md) | 定时任务、拒绝出信号的条件、邮件配置 |

## 旁支研究（均不接生产）

| 文档 | 结论 |
|---|---|
| [`systematic_half.md`](systematic_half.md) | 系统性成分的用途：分散化框架复现 —— **判负** |
| [`financial_structure_network.md`](financial_structure_network.md) | 财务结构网络：复现成立，交易应用 **判负** |
| [`dual_peer_effect.md`](dual_peer_effect.md) | 双重同伴效应：收益结论可复现，**机制解释不可复现** |

> 旁支研究一律附随机零基准。网络类结论若拿不出"优于随机图/随机网络"的证据，
> 一律判负——本项目此前多次因缺这把尺子而误判。
