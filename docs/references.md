# 参考资料与实现范围

[开放调优指南](open-tuning.md) 说明推荐的 `autotune` 入口；[RSI 指南](rsi.md) 和 [基础手册](guide.md) 说明旧版 `agent/workflow/tune` 的部署与评测；[架构文档](architecture.md) 说明模块及数据边界。这里记录设计依据和参考版本，便于区分借鉴思路、已实现功能和仍需后端适配的能力。

## 开放调优的引擎依据：vLLM-Ascend

开放模式由 LLM 负责诊断、策略选择和技术判断。每个有改动的候选在执行前，都会通过 MCP 或 A2A 研究接口重新查询 vLLM-Ascend 文档与实现，再由 LLM 审查相对当前参照的完整差异。研究记录和审查绑定候选及参照哈希；缺少来源、差异未覆盖、证据截断或审查仍有不确定性时，该候选不能执行。

控制器检查证据结构、引用来源类别、差异覆盖及哈希绑定，并不自行下载每个引用页面、核验其内容或证明版本兼容性。MCP/A2A 的回答和 LLM 的审查只是实验前的技术依据；真实启动、测量和独立复核仍决定最终结果。具体实现见 [`open_research.py`](../ascend_poc/open_research.py) 和 [`open_tuner.py`](../ascend_poc/open_tuner.py)。

以下官方页面于 2026-11-21 核查。`latest` 会变化，表中结论是该次阅读的范围；实际任务必须对照已安装版本、模型和设备查对应版本文档与源码，不能把这些入口当成固定版本的兼容性证明。

| 官方材料 | 已核查的条件与本项目边界 |
|---|---|
| [vLLM-Ascend 源码](https://github.com/vllm-project/vllm-ascend) / [安装说明](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html) | 部署依赖硬件与软件栈；仓库首页只提供入口，候选审查还需要具体文档和实现文件。CLI help 只证明参数拼写及形状，不能证明模型或硬件支持。 |
| [Feature Matrix](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/feature_matrix.html) | 功能存在兼容、部分兼容、不兼容和未知组合；需继续读对应功能的限制，不能将单项支持推成任意组合支持。 |
| [Graph Mode](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html) | 后端可能调整实际 graph mode；启动成功不证明请求的模式已生效。static kernel 还会增加启动编译成本。本文未声称本工具已自动读回所有实际生效设置或 profiler 证据。 |
| [Optimization and Tuning](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/optimization_and_tuning.html) | 当前说明明确 `TASK_QUEUE_ENABLE=2` 与 ACLGraph capture 冲突。此类限制应按实际版本查证；不能把所有性能开关一起打开作为通用方案。 |
| [Quantization](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ModelSlim 量化产物使用 `--quantization ascend`，LLM-Compressor 产物不必同样设置；还需匹配实际权重格式和硬件。改标志不会自动生成量化权重，本工具也没有集成权重量化流程。 |
| [Speculative Decoding](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html) | 方法、草稿模型、draft 长度与 graph 配置有前提条件。synthetic acceptance 会改变输出语义，只能作为明确标注的测试，不能充当普通模型推理收益。 |
| [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization/) | chunked prefill 的 token budget 会影响 TTFT、ITL 和吞吐，且与其他配置相互作用。上游 GPU 的示例值不是 Ascend 最优值或本项目固定搜索顺序。 |
| [vLLM bench serve](https://docs.vllm.ai/en/v0.28.0/cli/bench/serve/) | 该版本文档区分分位数、逐请求明细和带 TTFT/TPOT/E2EL 门槛的 goodput；实际可用参数以目标环境为准。指标必须经运行适配器接入，不能仅因文档列出就宣称当前模板会输出所有指标。 |

这些来源用于 LLM 理解候选，不替代客户目标。未写出的质量、精度或负载条件不会从上游示例补成客户硬门槛；改变模型、精度或负载时仍须报告实际条件，不能把不同工作点的得分差异描述为相同负载下的引擎加速。

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

## Agent 参考：ResearchBoost main

`main` 与 `bench` 的职责不同：`main` 是组织实际优化任务的 Agent 工作流，`bench` 是用于评估优化 Agent 的冻结任务集合。v0.5.0 同时参考两者，不能把基准包装本身当作 Agent 实现。

| 材料 | 固定版本或入口 |
|---|---|
| Agent 工作流 | [`ResearchBoost-Agent@1df28511f3add9101bdac0608ebcd776343a4bb4`](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/1df28511f3add9101bdac0608ebcd776343a4bb4) |
| 过程与评测边界 | [docs/methodology.md](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/1df28511f3add9101bdac0608ebcd776343a4bb4/docs/methodology.md) |
| 长程目标会话 | [docs/goal-mode.md](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/1df28511f3add9101bdac0608ebcd776343a4bb4/docs/goal-mode.md) |
| 实验经验 | [docs/lessons.md](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/1df28511f3add9101bdac0608ebcd776343a4bb4/docs/lessons.md) |

上游先由独立的 formulation 过程冻结任务与 judge，再让一个持久 coding-agent 会话按 profile、edit、time、check、log 迭代。它保留失败轮次，外部 supervisor 约束规则和事实，最终由会话外的 judge 裁决。其隔离层、代码编辑和训练等价性验证属于上游具体实现。

下表的持久记忆和动作会话对应旧版 `agent`；开放 `autotune` 使用本次任务内反馈，不包含该跨任务经验库。

| 上游思想 | 本项目旧版 `agent` 的实现 | 未声称具备的能力 |
|---|---|---|
| 长程目标与工具反馈 | LLM 逐步选择观察、研究、实验、反思、晋升和结束 | 没有接入上游 `/goal` 或 claw-orchestrator 会话 |
| 一次围绕一个改进假设 | 候选带 hypothesis、expected_effect 及原始测量 | 不自动实现任意源码或 kernel 补丁 |
| 保留失败与方法经验 | 同作用域案例及 profile 统计，供后续选择实验 | 历史成绩不能代替新环境验收，记忆不保证收益 |
| 外部独立最终裁决 | 可信执行器和 ABA 验证决定能否晋升 | 不等于已实现 Harbor 或 bubblewrap 沙箱 |
| 当前最好版本继续改进 | 已晋升冠军成为下一实验参照，记录代际谱系 | 不表示 LLM 权重、推理引擎源码或质量规则在自修改 |

这里的递归是已验证配置和经验反馈到后续实验。提升仍需在各模式规定的客户条件与评测协议下实测；新增工具动作、积累反思文字或增加运行轮数本身不是性能提升证据。

## 评测参考：RSI-Infra Bench

`bench` 分支提供长程优化中的冻结评测、独立裁决和实验记录方法，影响本项目的验收与审计部分。

| 材料 | 固定版本或入口 |
|---|---|
| `bench` 分支 | [`4c07eadf19363cb351688aee33ae06d3f176fd8d`](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/4c07eadf19363cb351688aee33ae06d3f176fd8d)，2026-11-18 |
| 评测协议 | [docs/eval-protocol.md](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/4c07eadf19363cb351688aee33ae06d3f176fd8d/docs/eval-protocol.md) |
| 具体计时实现 | [LLaVA bench.py](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/4c07eadf19363cb351688aee33ae06d3f176fd8d/tasks/llava/environment/workspace/bench/bench.py) |
| 对应博客 | [RSI-Infra Bench Preview](https://ai4ai-for-anything.github.io/blog/rsi-infra-bench/) |
| 博客源码快照 | [`febae5ba10640761bcb7baf0097574158b080715`](https://github.com/AI4AI-for-Anything/AI4AI-for-Anything.github.io/blob/febae5ba10640761bcb7baf0097574158b080715/blog/rsi-infra-bench/index.html)，2026-11-18 |

该博客对应 Bench 的评测设计；上面的 main 文档用于理解实际 Agent 的运行过程。

| 参考机制 | 本项目对应机制 |
|---|---|
| 固定任务与评测器 | 旧版采用客户契约、已验证策略目录和有限参数域；开放模式固定评测器与明确客户条件，未约束推理参数仍可探索 |
| 每轮自测、最终独立判定 | 探索与最终确认分开；只导出确认通过的配置 |
| 多项任务使用参考前后夹测 | 可配置的基线 A → 候选 C → 基线 B 复测 |
| 区分无收益、错误和无效测量 | `accepted` 与 `improved` 分开；参考异常为 `INVALID` |
| 冻结输入与完整日志 | 每轮指纹、声明文件 SHA-256、检查点绑定的事件链 |
| 有限优化预算 | 研究、搜索与最终复测共享总墙钟预算 |

旧版 ABA 按场景比较目标中位数；开放 ABA 保留逐场景硬门槛和参照漂移检查，但按客户选择的 `scenario_aggregation` 聚合目标裁决收益。开放模式遇到零或负目标值时使用归一化绝对差，不报告百分比或 speedup；初始参照不合格时，可通过同一候选三次独立启动建立合格基线，不能据此声称相对失败参照的加速。具体公式见 [开放调优指南](open-tuning.md) 与 [基础手册](guide.md)。这些机制借鉴前后参考和噪声门槛，但不逐项复刻上游所有训练任务的计时窗口、rank 聚合与漂移规则。

上游 N0 位级一致、N1 统计门槛属于训练等价性定义，不能代替客户业务准确率或质量评估。其 10 小时预算主要用于 agent 搜索；本工具的总预算包含最终确认，时间口径也不同。

博客关于测试覆盖不足的案例提醒：正确性检查通过并不单独证明真实负载受益。客户 trace、输出政策、缓存条件和保留测试集仍需有代表性；工具不会自动构造客户隐藏测试集。

## 统计与测量优化依据

旧版受控模式在 v0.6.0 增加成本观测、预算预检、可选探索早停、版本化经验作用域及独立多组 `aba_ci`。这些是本项目的实现选择，不是两个上游项目已在昇腾上验证的结论；当前开放 `autotune` 不支持 `aba_ci` 或旧版跨任务记忆，不能将下述统计区间归给开放模式的普通 ABA。

| 主源 | 用于检查的假设与限制 |
|---|---|
| [NIST：单总体均值的 t 检验](https://www.itl.nist.gov/div898/handbook/prc/section2/prc22.htm) | 小样本的 t 分位数和自由度；不能固定使用 `1.96` 冒充任意样本量的区间 |
| [NIST：双样本 t 检验](https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm) | 配对与非配对、方差及目标差异的区分；本实现采用 block 配对效果，未把它命名为 Welch 检验 |
| [NIST：Bonferroni 方法](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm) | 对预定有限比较分配错误率；单次区间不能覆盖任意多次任务重跑 |
| [NIST：自相关测量的不确定度](https://www.nist.gov/programs-projects/uncertainty-analysis-autocorrelated-measurement-data) | 相关重复不能直接套用假设独立的普通不确定度公式 |
| [NIST：顺序趋势与独立性](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd444.htm) | 时间漂移或趋势需要先识别，不能只将观测混合后扩大区间 |

旧版 `aba_ci` 按每场景、每独立 ABA block 的方向归一化 log-ratio 计算单侧 Student t 下界，并按场景数与预定比较上限分配错误率。原始客户目标、逐重复硬门槛及原 ABA 裁决继续保存；块内和跨块参照漂移仍能使比较无效。区间依赖独立 block 与近似正态的 block 效果，代码不自动证明这些假设。

成本预检采用观测与操作者粗估值，不声称预测真实 NPU 时长；探索早停仅删除已不可能达标的剩余探索测量。共享 ABA 参照、分阶段重新加载模型、跨检查摘要缓存和弱化文件校验未作为此版本的默认优化。测试使用数值案例、协议夹具和合成流程，不能推出现场误晋升率或节省时间百分比。

## 归属与复现声明

本项目中的 Agent、经验、回放、影子建议和验收逻辑是针对本项目接口独立实现的，没有直接复制上述项目的 engine、router、Trie 重放器或训练 judge。此文档记录方法来源，不表示二者为本项目背书。

如果后续直接移入第三方源码，应另外核查对应文件的许可证、版权头与 NOTICE，并保留所需声明，不能仅凭仓库首页的许可证概括所有子目录。

当前自动化测试覆盖 Python 逻辑、本地 HTTP/SSE 协议和合成优化流程。真实昇腾硬件、客户数据集质量和付费模型 API 的验证，需要在相应部署环境执行。所有来源的排行榜或加速倍数均未被当作本工具的实测成绩。
