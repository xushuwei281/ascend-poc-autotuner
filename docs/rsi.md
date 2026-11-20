# Inference RSI Agent 使用指南

先看 [图解调优](tuning-explained.md) 了解参数与策略，再按 [从零复现](quickstart.md) 跑通演示。本页解释 Agent 的具体操作、限制和证据规则。

大模型每步选择工具，读取工具结果，再决定下一步。候选通过独立验证后可以晋升为冠军，后续实验继续以它为参照；实验经验保留到下一次运行，但不能代替新的实测。v0.6.0 增加成本感知的实验预算、可选探索早停、多组置信区间验证和经验对照。

这是配置与策略层的递归改进。可改变的是已批准策略及有限参数、实验顺序和经验；客户模型/精度约束、请求负载、质量线、评分与评测脚本保持固定。不能把晋升次数解释为 LLM 智力或模型权重已提升。

1. [快速体验](#1-快速体验)
2. [客户环境接入](#2-客户环境接入)
3. [配置与工具](#3-配置与工具)
4. [实验、晋升与回退](#4-实验晋升与回退)
5. [经验与递归改进](#5-经验与递归改进)
6. [预算、恢复与报告](#6-预算恢复与报告)
7. [适用边界与排障](#7-适用边界与排障)

## 1. 快速体验

需要 Linux 和 Python 3.10+；运行时不依赖第三方 Python 包。在仓库根目录执行：

```bash
python3 -m ascend_poc agent-demo --output runs/rsi-demo
python3 -m ascend_poc agent-report --output runs/rsi-demo
```

完整体验可选的独立区组统计模式：

```bash
python3 -m ascend_poc agent-demo --verification aba_ci --output runs/rsi-ci-demo
python3 -m ascend_poc agent-report --output runs/rsi-ci-demo
```

两种演示均为本地合成数据；CI 演示运行更多独立服务启动，不代表硬件统计假设已经成立。

演示不需要 NPU 或外部 API key，使用本地服务与协议替身。它验证工具交互和实验链路，所有性能数字均为合成数据。再次运行使用新目录。安装和环境检查见 [基础手册](guide.md#1-环境与安装)。

`agent_report.md` 是主报告。没有发生有效晋升时不导出冠军文件；运行结束、LLM 说“成功”或探索阶段得分高，都不能替代最终验证。

## 2. 客户环境接入

### 准备客户契约与策略目录

先完成 [基础手册的客户接入](guide.md#2-接入客户-poc)：手工跑通推理服务、性能与质量脚本，固定软硬件版本、模型、卡数、数据和验收矩阵。

Agent 复用原有两层配置：

- **客户契约**定义固定场景、质量脚本、目标与门槛，也包含规划 LLM 的 API 配置。
- **工作流目录**定义允许试验的 profile、有限参数域、兼容性证据、重复次数和 ABA 门槛。

首个 profile 必须保持原契约基线。只有启用并由操作者标记 `verified` 的 profile 可运行；Agent 不能把研究回答直接变成兼容性证据，也不能生成任意启动命令。

### 增加 Agent 配置

已在仓库根目录准备好 `customer-workflow.json` 后：

```bash
cp examples/inference-rsi.json customer-agent.json
python3 - <<'PY'
import json
from pathlib import Path
path = Path("customer-agent.json")
config = json.loads(path.read_text())
config["workflow"] = "customer-workflow.json"
config["memory_file"] = "runs/customer-rsi-memory.json"
config["goal"] = "保持客户负载和质量约束，满足全部门槛后提高最坏场景吞吐。"
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
```

`goal` 用于表达意图，真正的目标、方向和硬门槛仍来自客户契约。如果文字目标与契约冲突，先由操作者修正契约后新开实验，不能让 Agent 在过程中放宽阈值。

### 配置 API 并启动

```bash
export MODEL_PATH=/absolute/path/to/approved-model
read -rsp 'Planner API key: ' POC_LLM_API_KEY
export POC_LLM_API_KEY
python3 -m ascend_poc agent-validate --config customer-agent.json
python3 -m ascend_poc agent \
  --config customer-agent.json --output runs/customer-rsi --workdir "$PWD"
```

LLM 使用工作流所引用契约的 `llm` 配置，支持 OpenAI-compatible Chat Completions；密钥、JSON mode 和超时设置见 [API 说明](guide.md#5-大模型-api-与提案格式)。程序不会自动加载 `.env`。实际 API 调用按供应商规则计费。

DeepWiki 连接使用工作流的 `research` 配置。可直接使用 MCP，也可先启动 `research-agent` 经本机 A2A 桥查询，见 [研究服务配置](guide.md#4-策略与研究配置参考)。

研究请求包含所选主题、允许分享的环境信息、当前冠军和近期实验的受限诊断。未达标指标与负载窗口帮助定位问题；原始提示词、生成内容、启动命令和进程日志不会自动作为查询上下文。资料仍是待验证建议，不能批准新策略。

## 3. 配置与工具

Agent 配置示例：

```json
{
  "schema_version": 1,
  "name": "Customer inference RSI",
  "goal": "保持固定客户验收条件，寻找可复验的性能提升。",
  "workflow": "customer-workflow.json",
  "memory_file": "runs/customer-rsi-memory.json",
  "settings": {
    "max_steps": 40,
    "max_experiments": 8,
    "max_wall_time_s": 7200,
    "experiment_timeout_s": 900,
    "max_llm_calls": 80,
    "max_research_calls": 12,
    "max_consecutive_tool_errors": 3
  }
}
```

`workflow` 相对 Agent 配置文件解析；`memory_file` 的相对路径也以配置文件所在目录为基准。客户命令中的工作目录则由 `--workdir` 指定。不要混淆这两种路径。

| 设置 | 含义 |
|---|---|
| `max_steps` | 限制 Agent 工具决策步数，避免无休止观察或反思 |
| `max_experiments` | 限制完整候选实验次数，不是底层 benchmark 请求数 |
| `max_wall_time_s` | 本次 Agent 的总墙钟预算，包含研究、规划、测量和确认 |
| `experiment_timeout_s` | 单实验预算上限；同时受总剩余时间限制 |
| `max_llm_calls` | 规划 API 的 HTTP 尝试上限，重试也计数 |
| `max_research_calls` | 本次研究调用上限 |
| `max_consecutive_tool_errors` | 连续工具错误达到上限时停止，避免反复执行无效操作 |

示例数值不是客户统一推荐值。大模型加载、多个场景和最终重复会消耗大量时间；单实验需要能容纳参照、候选及最终 ABA 全部测量，不能只按一次压测估算。

LLM 每步返回一个 JSON 对象：

```json
{"tool": "inspect", "arguments": {}}
```

这是 Chat Completions 内容中的应用层 JSON 协议，不要求供应商支持原生 function calling。控制器只执行注册工具，严格验证名称和参数；模型输出的其他命令不会作为 shell 运行。

| 工具 | 参数 | 作用 |
|---|---|---|
| `inspect` | 无 | 查看当前代、历史实验、可选策略和剩余预算 |
| `recall` | 无 | 读取相同作用域的历史案例与策略统计 |
| `research` | `topic` | 在允许的研究主题中查询 DeepWiki |
| `experiment` | `profile_id`、`parameters`、`hypothesis`、`expected_effect` | 用完整合法参数运行一个候选实验 |
| `reflect` | `experiment_id`、`lesson` | 将经验关联到已有实验，不修改其测量事实 |
| `promote` | `experiment_id` | 请求晋升，控制器重新核对独立证据及当前代 |
| `rollback` | `reason` | 将冠军导出与下一轮参照退回上一代 |
| `finish` | `reason` | 说明达成目标、无收益或其他停止原因并结束 |

`hypothesis` 和 `expected_effect` 用于描述“为什么改、预计哪个指标变好”。`lesson` 是模型的总结，可能有错；它不覆盖客户门槛、验证结果或测量值。`research.topic` 只接受 `scheduling`、`cache`、`prefill`、`decode`、`parallelism`、`quantization`、`speculation`，不是可执行命令或任意 URL。

## 4. 实验、晋升与回退

一次 `experiment` 使用子工作流测试两个探索配置：**当前冠军作为参照**，以及 Agent 提出的候选。只有候选入围，才进入独立复核；参照胜出时不再给参照自身做一组无法让该候选晋升的 ABA。通用 `workflow` 仍可按自身目录确认基线。

`aba` 对候选执行参照 A → 候选 C → 参照 B，每组使用固定场景及最终重复次数。可选 `aba_ci` 执行多组相互独立启动的完整 ABA，保留逐组和跨组漂移保护，并增加置信下界条件。设置、成本和统计假设见 [验证协议](guide.md#6-预算硬门槛与独立复测)。

候选探索失败、参照胜出、预算预检跳过或候选没有完成最终确认，都不能根据探索成绩晋升。“未确认”与“已完成独立验证但失败”应分别解读。晋升必须同时满足：

1. 候选通过所有客户场景、每次重复的硬门槛。
2. ABA 参照测量完整且漂移在限制内。
3. 候选的 `accepted` 与 `improved` 都成立，收益超过工作流声明的噪声/改进阈值。
4. 实验使用的参照与当前冠军代仍一致，不能将对旧代的结果套到新代。
5. 本次配置快照、运行指纹、声明冻结文件和实验记录保持有效。

Agent 的自由判断不能替代这些条件。验证模式在运行前冻结；`aba_ci` 还要求全部预定 block 完整、置信下界过线，不能在看到结果后挑选有利子集。其 `ci_max_comparisons` 必须至少覆盖 Agent 的 `max_experiments`，不能以每个子实验重新开始为由重置比较次数上限。

晋升后保存冠军谱系，并将新配置作为下一次实验参照。客户契约仍然固定，变化的是比较对象。后续的改进率是相对当轮参照测得，不能简单将各代百分比连乘成“相对原始基线的实测加速”。要报告该数字，应另外用原始基线和最终冠军完成同口径复测。

`generation` 在晋升或回退时都单调递增；冠军记录包含 `profile_id`、`parameters` 和 `experiment_id`。`rollback` 只回退控制器的冠军配置、导出配置和下一实验参照。它不停止生产服务、不修改在线节点角色、不迁移 KV，也不代表旧配置在当前时间重新通过了测试。

## 5. 经验与递归改进

经验文件保存已测案例、profile 统计以及按 profile、参数名和值分组的边缘计数，帮助下一次 Agent 选择实验顺序。不同类型的值单独分组，不将 `true` 与 `1` 混在一起。这些计数是已有实验的描述，不是参数的因果作用或校准过的成功概率；其他参数、候选选择和参照变化都可能影响结果。

探索阶段的 `observed_feasible`、`violated_metrics` 与独立裁决分开保存；`observed_rejected_count` 统计探索不达标，不能冒充最终复核失败。即使候选因 SLO 失败或早停未进入 ABA，也会留下负例。反思与无收益同样保留，但不能变成已通过验收的配置。

v0.6.0 的 namespace 使用版本化测量作用域：

| 影响经验作用域 | 仅影响搜索组织，不单独隔离经验 |
|---|---|
| 服务与测试命令/环境、场景、质量要求、目标、参数域及约束 | 名称与描述、规划 LLM 设置、研究连接设置 |
| 运行指纹、声明冻结文件 | 搜索次数、总搜索时间、入围数量 |
| `repetitions`、`final_repetitions`、验证模式与门槛及 CI 参数 | 成本估算与探索优化开关 |

旧版 full-workflow namespace 不自动迁移到新作用域，也不通过猜测宣布兼容。旧案例仍留在原文件，但不会作为新版匹配经验出现。经验复用与 `--resume` 是两套规则：恢复同一次任务仍要求完整配置、运行模式与原始截止时间一致。

即便作用域相同，历史案例也只是**下一次实验的候选建议**。旧分数不会直接成为本次冠军，旧经验不跳过 ABA，旧 `verified` 声明也不会自动证明新机器兼容。运行指纹依赖可观测环境与操作者声明，不是所有硬件、权重目录和外部服务的完整认证。

递归改进包含两个实际反馈：

- **配置反馈**：经验证的新冠军成为下一轮实测参照。
- **经验反馈**：案例和策略统计影响之后的工具与候选选择。

本版没有自动训练 LLM、修改控制器源码或生成任意新 kernel。记忆变多也不保证搜索更快；可以用相同协议、预算与独立输出目录做对照：

```bash
python3 -m ascend_poc agent --config customer-agent.json \
  --output runs/customer-with-memory --workdir "$PWD"
python3 -m ascend_poc agent --config customer-agent.json \
  --output runs/customer-without-memory --workdir "$PWD" --ignore-memory
```

`--ignore-memory` 禁止读取与写入持久经验；本次运行自己的实验历史仍参与决策。它不会禁用反思，也不会把本次已经观察到的失败隐藏起来。恢复此任务必须继续带相同开关。正式比较应预先固定经验快照，并让每组使用隔离文件；交替运行次序及记录现场环境，避免把温度变化或 API 波动当成经验收益。

报告的 `milestones` 分别记录候选首次探索达标、首次独立验收和首次晋升，每项保存已启动实验数 `experiments` 和已用墙钟秒 `elapsed_s`：

| 键 | 事件 |
|---|---|
| `first_observed_feasible` | Agent 提出的候选首次观察达标，不是子实验参照达标 |
| `first_independent_acceptance` | 候选首次通过完整独立验收，未必证明提升 |
| `first_promotion` | 候选首次实际晋升为新冠军 |

尚未发生的事件保持未达到，不能按零耗时计入平均数。除了次数，也比较总墙钟、达到客户 SLO 的比例及失败原因。

经验和研究内容均按不可信输入处理。它们不能授权新命令、改写评测器或放宽质量线。不要将客户原始 prompt、密钥或敏感业务内容写进目标、假设和经验文本。

每次实验还向 Agent 返回派生的瓶颈分析、未达标指标，以及可用的缓存/未缓存 token 和负载窗口摘要；缺少观测时保持未知。长程运行会压缩发送给 LLM 的历史，优先保留近期实验和当前冠军，并截断研究文本及经验。完整记录仍保存在本地，客户契约与验收条件不会被历史摘要替换。

## 6. 预算、恢复与报告

总预算使用启动时确定的绝对截止时间，暂停和等待同样消耗时间。Agent 步数、LLM、研究和实验各有上限；达到任一适用上限时停止继续相应动作，不能通过重启清零。

工作流 `budget` 可启用 `confirmation_reserve`，按已观测启动/测试成本动态预留确认时间并在确认组开始前预检；`early_stop` 只用于探索已出现硬门槛失败的配置。两者旧配置默认关闭，具体字段见 [基础手册](guide.md#6-预算硬门槛与独立复测)。估算不是完成时限保证：未知成本、慢启动或环境波动仍可能耗尽预算。`experiment_timeout_s` 不会自动放大；`aba_ci` 的单个候选仅确认就需要 `3 × ci_blocks` 次完整测量，应据此设置时限。

新版常规工作流示例显式开启这两项优化，Agent 继承其设置。CI 版本可参考 `examples/inference-rsi-ci.json`；用新目录和足够预算启动，不要直接把正在运行的 `aba` 任务改成 CI 模式后恢复。

```bash
python3 -m ascend_poc agent \
  --config customer-agent.json --output runs/customer-rsi \
  --workdir "$PWD" --resume
python3 -m ascend_poc agent-report --output runs/customer-rsi
```

恢复必须使用原配置和有效事件记录。尚未完成的实验会按保存状态恢复处理；缺少完整验证的候选不能因进程重启而晋升。修改配置、代码、环境或冻结输入后使用新输出目录。

| 产物 | 用途 |
|---|---|
| `agent_report.md` | 步骤计数、实验结论、冠军谱系及停止原因 |
| `agent_state.json` | 当前状态、预算计数和待完成实验 |
| `agent_manifest.json` | 本次 Agent 的输入与完整性记录 |
| `events.jsonl` | 完整工具步骤、Agent 事件及检查点链 |
| `experiments/` | 每次子工作流的配置、原始测量、成本/早停记录、分析与独立验证 |
| `champion.json` | 已晋升配置的身份、参数及对应证据 |
| `champion-contract.json` | 已晋升配置的可复现契约 |
| `memory_file` 指定文件 | 跨任务经验与策略统计，建议存放于被忽略的 `runs/` 下 |

子工作流的 `best.json` 只表示该子工作流有最终接受的配置，未必有显著提升；它不等于 Agent 的冠军晋升。旧版子实验或通用工作流的赢家还可能是参照。Agent 的事件链与子工作流的事件链是不同层级。`agent-report` 面向 Agent 根目录；`report` 仍面向某一个原有 workflow 或 tune 输出目录。

交付应包含原始客户契约、软件/硬件说明、质量结果、冠军配置、全部测量及独立确认记录。不要只提供一个最好吞吐数字或反思文字。所有日志、经验和客户配置默认留在客户指定环境，避免提交到源码仓库。

`agent` 退出码：`0` 有当前有效的已确认冠军；`2` 没有有效晋升，即使原始基线达标；`1` 配置、完整性或执行异常；`130` 被中断。

## 7. 适用边界与排障

| 问题 | 处理 |
|---|---|
| 校验失败 | 先校验工作流与客户契约；检查相对路径、策略域和 API 设置 |
| 可选策略只有基线 | 现场验证其他策略后再启用，不要伪造兼容性证据 |
| 工具调用持续报错 | 查看报告中的工具名与错误，修正接线或合法参数；不要扩大执行权限来掩盖问题 |
| 实验达标但不能晋升 | 核对候选是否最终确认、是否有显著收益、参照代是否已变化 |
| 没有冠军文件 | 没有满足晋升条件的候选；初始参照不自动算一次成功晋升 |
| 有历史经验却仍然压测 | 历史经验只用于选候选，本次成绩必须重新测量 |
| ABA 超时或无效 | 为模型加载、完整矩阵及全部确认 block 留足预算，排查机器波动和输入变化 |
| 恢复被拒绝 | 核对原配置、指纹与记录；环境改变后新开实验 |
| 想实时调整 P/D | `adaptive-plan` 仍为 shadow；必须先实现并验证 Ascend 后端控制协议 |

`workflow` 和 `tune` 仍可单独运行。Agent 可以调用它们已有的可信执行和验收逻辑，但不将任何一次模型推理当作独立 judge。

本次设计主要借鉴 EvolvingInfra 的负载观察，以及 ResearchBoost 的长程目标、逐轮实验记录与外部最终裁决。实现对应、固定提交和未移植能力见 [参考来源](references.md)。CPU 测试和本地 HTTP fixture 能证明控制逻辑，不能证明真实 Ascend 性能、客户质量或所有 API 服务商兼容性。
