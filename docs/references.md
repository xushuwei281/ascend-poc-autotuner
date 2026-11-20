# 参考资料与实现范围

[使用手册](guide.md) 说明如何运行；[架构文档](architecture.md) 说明模块及数据边界。这里记录设计依据和固定参考版本，便于区分借鉴思路、已实现功能和仍需后端适配的能力。

## 主要参考：EvolvingInfra

**Evolving Infra: Agent-Adaptive LLM Serving at Scale** 是本项目动态 Agent 工作负载方向的主要架构参考。

| 材料 | 固定版本或入口 |
|---|---|
| 仓库 | [`mit-han-lab/EvolvingInfra@8115657fae5d2c78b37bcbfa27fe9d0e5d3fe212`](https://github.com/mit-han-lab/EvolvingInfra/tree/8115657fae5d2c78b37bcbfa27fe9d0e5d3fe212)，2026-11-19 |
| 论文 | [arXiv:2611.12304v2](https://arxiv.org/html/2611.12304v2)，2026-11-12 |
| 项目主页 | [inference-rsi.com](https://inference-rsi.com/) |
| 成本模型与 planner | [planner/README.md](https://github.com/mit-han-lab/EvolvingInfra/blob/8115657fae5d2c78b37bcbfa27fe9d0e5d3fe212/evolving_infra/planner/README.md) |
| 角色切换协议 | [docs/dynamic-pd.md](https://github.com/mit-han-lab/EvolvingInfra/blob/8115657fae5d2c78b37bcbfa27fe9d0e5d3fe212/docs/dynamic-pd.md) |
| Agent KV 生命周期 | [docs/kv-retention.md](https://github.com/mit-han-lab/EvolvingInfra/blob/8115657fae5d2c78b37bcbfa27fe9d0e5d3fe212/docs/kv-retention.md) |
| 轨迹回放 | [benchmarks/replay/README.md](https://github.com/mit-han-lab/EvolvingInfra/blob/8115657fae5d2c78b37bcbfa27fe9d0e5d3fe212/benchmarks/replay/README.md) |

EvolvingInfra 将计算角色与 KV 存储分开，利用定制引擎、router 和共享 host-memory KV pool 降低在线角色变化的代价。运行时 planner 根据实际流量拟合成本模型，Agent 分支和上下文管理信号则帮助判断 KV 是否仍会复用。LLM / coding agent 主要承担部署、接入、诊断和开发。

### 本项目采用的部分

| 思路 | 当前实现 | 使用限制 |
|---|---|---|
| 区分实际 prefill 工作量 | 记录 input、cached、uncached 和实际 output tokens | 缺失 cache usage 保持未知，不能从 prompt 重叠推定命中 |
| 保留 Agent 负载结构 | 固定请求 DAG、分支依赖、外部等待、上下文版本观测 | 不执行真实工具，不把新输出回填至后续记录 |
| 成本模型驱动资源建议 | 根据已提供系数和等长窗口比较 P/D 容量 | 仅 `shadow`；不自动拟合调度 trace，不验证校准来源 |
| 避免短暂波动触发调整 | 标准误差门槛、相对收益下限、冷却与缺测检查 | 只分析传入历史窗口，不检查样本距当前时间的年龄 |
| Agent 可接手维护 | 根目录 `AGENTS.md`、可运行示例和测试入口 | 不代表所有部署环境或后端功能已经验证 |

完整因果注意力中，未缓存 token 数 `u` 和缓存 token 数 `c` 对应的 query–key pairs 代理为 `u*c + u*(u+1)/2`。本项目保留这一观测思路，但不把它当成所有注意力架构、多模态模型或设备上的直接耗时。

影子建议器的 prefill 模型包含固定开销、uncached tokens、attention pairs 和恢复 tokens；decode 模型包含固定开销、batch 和期望专家覆盖率。系数应来自相同模型、硬件与并行方式。示例中的系数是合成数据，没有沿用 Hopper 系数或将其作为 Ascend 校准结果。

### 尚未移植的后端能力

| 能力 | 需要补充的实现与证据 |
|---|---|
| 无重启 P/D 角色切换 | 目标推理后端的明确运行时接口，以及卡型、模型和版本验证 |
| 在途请求 drain / 迁移 | 不丢请求、不重复输出、超时可恢复的协议与测试 |
| 跨节点 KV 保留和传输 | NPU、互联、传输层和 KV 格式的兼容性与正确性 |
| Agent 感知驱逐 | 分支引用、上下文替换、释放确认及后端 cache 实现 |
| 在线决策提交 | 状态版本检查、响应不明确时读回核验、切换后观测与回退 |

`agent_id`、`branch_id` 和 `context_version` 在本项目中用于观测，不能触发这些后端动作。添加上游同名请求头，也不会让普通 vLLM 服务自动具备 KV 保留协议。

### 版本与实验口径

固定快照的 `docs/dynamic-pd.md` 说明已测试脚本式角色切换，planner 自己决策触发的路径尚未端到端运行验证。该说明限定了这份代码快照的已测范围，不用于否定论文的全部实验。

同一快照中，架构说明与较新的 retention 文档、启动默认值对“已测配置”和“默认启用”存在差异。部署时应记录提交、实际配置和测试结果，不能仅凭 README 或默认值推断完整能力。

论文的性能结果属于指定模型、NVIDIA GPU 和实验条件。上游固定缓存返回时间表也来自特定轨迹；这些倍数、等待时间和校准系数都不是本工具的性能承诺或客户通用默认值。

## 评测参考：ResearchBoost / RSI-Infra Bench

ResearchBoost 提供长程优化中的冻结评测、独立裁决和实验记录方法，主要影响本项目的验收与审计部分。

| 材料 | 固定版本或入口 |
|---|---|
| `bench` 分支 | [`4c07eadf19363cb351688aee33ae06d3f176fd8d`](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/4c07eadf19363cb351688aee33ae06d3f176fd8d)，2026-11-18 |
| 评测协议 | [docs/eval-protocol.md](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/4c07eadf19363cb351688aee33ae06d3f176fd8d/docs/eval-protocol.md) |
| 具体计时实现 | [LLaVA bench.py](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/4c07eadf19363cb351688aee33ae06d3f176fd8d/tasks/llava/environment/workspace/bench/bench.py) |
| 对应博客 | [RSI-Infra Bench Preview](https://ai4ai-for-anything.github.io/blog/rsi-infra-bench/) |
| 博客源码快照 | [`febae5ba10640761bcb7baf0097574158b080715`](https://github.com/AI4AI-for-Anything/AI4AI-for-Anything.github.io/blob/febae5ba10640761bcb7baf0097574158b080715/blog/rsi-infra-bench/index.html)，2026-11-18 |

该博客是本次采用的 Bench 参考，不是较早的 `ydove0324.github.io/hx_resume/blog/researchboost/` 项目介绍。

| 参考机制 | 本项目对应机制 |
|---|---|
| 固定任务与评测器 | 客户契约、已验证策略目录和有限参数域 |
| 每轮自测、最终独立判定 | 探索与最终确认分开；只导出确认通过的配置 |
| 多项任务使用参考前后夹测 | 可配置的基线 A → 候选 C → 基线 B 复测 |
| 区分无收益、错误和无效测量 | `accepted` 与 `improved` 分开；参考异常为 `INVALID` |
| 冻结输入与完整日志 | 每轮指纹、声明文件 SHA-256、检查点绑定的事件链 |
| 有限优化预算 | 研究、搜索与最终复测共享总墙钟预算 |

本项目的 ABA 是逐场景比较推理目标中位数，具体公式见使用手册。它借鉴前后参考和噪声门槛，但不逐项复刻上游所有训练任务的计时窗口、rank 聚合与漂移规则。

上游 N0 位级一致、N1 统计门槛属于训练等价性定义，不能代替客户业务准确率或质量评估。其 10 小时预算主要用于 agent 搜索；本工具的总预算包含最终确认，时间口径也不同。

博客关于测试覆盖不足的案例提醒：正确性检查通过并不单独证明真实负载受益。客户 trace、输出政策、缓存条件和保留测试集仍需有代表性；工具不会自动构造客户隐藏测试集。

## 归属与复现声明

本项目中的回放、影子建议和验收逻辑是针对本项目接口独立实现的，没有直接复制上述项目的 engine、router、Trie 重放器或训练 judge。此文档记录方法来源，不表示二者为本项目背书。

如果后续直接移入第三方源码，应另外核查对应文件的许可证、版权头与 NOTICE，并保留所需声明，不能仅凭仓库首页的许可证概括所有子目录。

当前自动化测试覆盖 Python 逻辑、本地 HTTP/SSE 协议和合成优化流程。真实昇腾硬件、客户数据集质量和付费模型 API 的验证，需要在相应部署环境执行。所有来源的排行榜或加速倍数均未被当作本工具的实测成绩。
