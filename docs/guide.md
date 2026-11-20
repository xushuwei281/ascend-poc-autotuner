# 完整使用手册

第一次接触推理调优，先看 [图解调优](tuning-explained.md)；想先跑出一份演示报告，跟着 [从零复现](quickstart.md) 操作。本页是接入真实测试和查配置字段的详细手册。

本项目面向昇腾售前 PoC：在客户认可的模型、硬件、数据、精度和请求分布下，寻找满足全部验收门槛的推理配置。推荐入口是 `workflow`：测基线、分析结果、查询 DeepWiki、调用大模型选择已验证策略、执行候选、独立复测，再导出成绩与复现配置。

它包含服务启动与测试的慢循环、Agent 请求轨迹回放、负载统计和 P/D 容量预测。P/D 预测只输出建议，不修改运行中的推理引擎。实现结构见 [架构说明](architecture.md)，与 EvolvingInfra、ResearchBoost 的对应关系见 [参考与边界](references.md)。

1. [环境与安装](#1-环境与安装)
2. [接入客户 PoC](#2-接入客户-poc)
3. [客户契约配置](#3-客户契约配置参考)
4. [策略与研究配置](#4-策略与研究配置参考)
5. [大模型 API](#5-大模型-api-与提案格式)
6. [预算、硬门槛与复测](#6-预算硬门槛与独立复测)
7. [Agent 回放与遥测](#7-agent-轨迹回放与遥测)
8. [P/D shadow 预测](#8-pd-容量预测shadow-模式)
9. [结果、恢复与交付](#9-结果恢复与交付)
10. [CLI、排障与验证](#10-cli排障与验证边界)

## 1. 环境与安装

控制器需要 Linux、Python 3.10+；运行时仅使用 Python 标准库。安装构建需要 `setuptools>=68`。CPU demo 不需要 NPU、模型权重、外部 API 或 vLLM。

真实测试还需要客户提供并验证：昇腾驱动、CANN、torch/torch-npu、匹配版本的 vLLM/vLLM-Ascend、模型及 tokenizer、固定测试数据和性能/质量评测脚本。控制器不会安装或升级这些组件。

在仓库根目录直接运行源码：

```bash
python3 --version
python3 -m ascend_poc --help
python3 -m ascend_poc doctor
```

需要虚拟环境和命令行入口时，从当前仓库安装；这里不假设 PyPI 已发布同名版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ascend-poc --help
```

`doctor` 输出 Python、系统、相关包版本以及可获得的 `npu-smi info`。它不验证完整软硬件兼容性，也不调用模型 API。

### 60 秒 CPU 演示

```bash
python3 -m ascend_poc workflow-demo --output runs/first-demo
python3 -m ascend_poc report --output runs/first-demo
```

使用新的输出目录。演示在本机模拟 MCP、A2A、LLM、服务与指标，通常可在一分钟左右完成；实际耗时取决于机器。查看 `runs/first-demo/workflow_report.md`、`best.json` 和 `events.jsonl`。演示证明控制流程可运行，所有性能数字都是合成数据。

## 2. 接入客户 PoC

### 第一步：冻结验收口径

先与客户确定模型/精度、卡型/卡数/拓扑、请求到达方式、并发、输入输出长度、缓存及预热政策、测试持续时间、错误分母、质量集和指标定义。TTFT、TPOT、ITL、单请求延迟、任务延迟不能相互替代。

记录镜像或软件版本、模型 revision、tokenizer 与数据身份。不同候选必须使用同一套验收矩阵；不能降低输出长度、跳过失败请求或拼接多个配置的最好单项成绩。

### 第二步：准备固定测试适配器

先手工跑通客户的性能和质量测试。适配器必须以非零退出码报告执行失败，并把有限数值写入新的 JSON 文件。例如性能结果：

```json
{
  "output_tokens_per_second": 840.5,
  "p99_ttft_ms": 1800,
  "p99_tpot_ms": 42,
  "completed": 1000,
  "failed": 0,
  "error_rate": 0
}
```

质量结果单独写入，例如 `{"quality_score": 0.97}`。这些数字仅说明格式；质量函数、数据和及格线必须由客户确定。质量结果不能覆盖性能文件中的同名指标。

复制可编辑模板：

```bash
cp examples/customer-template.json customer-contract.json
cp examples/ascend-workflow.json customer-workflow.json
python3 - <<'PY'
import json
from pathlib import Path
path = Path("customer-workflow.json")
config = json.loads(path.read_text())
config["contract"] = "customer-contract.json"
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
```

[customer-template.json](../examples/customer-template.json) 的 `/opt/customer/run_performance.py` 和 `run_quality.py` 是待替换的客户脚本路径，仓库不提供它们。[vllm-ascend.json](../examples/vllm-ascend.json) 是合成随机请求性能模板，没有质量评测；不能直接用于质量验收。

### 第三步：分别填写契约与策略目录

| 客户契约 `customer-contract.json` | 工作流 `customer-workflow.json` |
|---|---|
| 固定硬件资源标识、服务地址、模型与上下文上限 | `contract` 改为 `customer-contract.json` |
| 性能/质量测试、缓存政策、门槛和优化目标 | 选择 MCP 或本机 A2A 研究通道 |
| 有限参数域、基线值和条件约束 | 声明允许尝试的策略与兼容性证据 |
| 大模型 API 地址、模型 ID、密钥环境变量名 | 设置全局预算、最终验证和文件哈希 |

模板中的两卡、设备 `0,1`、端口 `8000`、服务模型名 `poc-model`、上下文 `8192` 和所有 CLI 参数都需要与客户环境核对。模型 API 的 `model` 填供应商实际提供的 ID，不要沿用占位符。

`profiles[0]` 必须保持契约原始策略，且启用并标为 `verified`。替换模板中的兼容性说明，写入硬件、模型、软件版本和已完成的基线验证记录；模板的文字不是验证证据。

其他策略只有同时满足 `enabled: true` 和 `compatibility.status: "verified"` 才能执行。先验证 APC、图模式、chunked prefill、推测解码或量化在具体版本上的支持，再开放相应 profile。大模型不能自行把策略改成已验证。

### 第四步：设置环境、校验与执行

```bash
export MODEL_PATH=/absolute/path/to/approved-model
read -rsp 'Planner API key: ' POC_LLM_API_KEY
export POC_LLM_API_KEY
python3 -m ascend_poc validate --config customer-contract.json
python3 -m ascend_poc workflow-validate --config customer-workflow.json
python3 -m ascend_poc workflow \
  --config customer-workflow.json \
  --output runs/customer-poc --workdir "$PWD"
```

校验命令只检查结构和逻辑，不启动服务、不查询 API、不认证 profile 证据。先用小预算完成接线检查，再用新的输出目录执行正式实验。

服务地址必须专用且端口空闲。所有服务和 wrapper 必须在前台运行；控制器按进程组启动、终止自己拥有的服务，每轮结束清理。不要使用 `docker -d`、后台 shell 或未受控 SSH 服务替代生命周期管理。

`resource_id` 是跨输出目录的本机资源锁标识；使用同组 NPU 的作业应填写同一值。锁不能阻止其他不遵守此协议的进程占卡。

## 3. 客户契约配置参考

配置是严格 JSON：重复键、未知结构字段、NaN/Infinity、非法参数值会被拒绝。下表覆盖契约支持的全部顶层字段；标记“默认”的字段可省略。

| 字段 | 类型、默认值与含义 |
|---|---|
| `schema_version` | 必须为整数 `1` |
| `name`、`description` | 名称必填；描述可选 |
| `resource_id` | 字符串，默认 `ascend-default` |
| `context` | 默认 `{}`；非敏感硬件、模型、负载背景，会进入规划上下文 |
| `server` | 必填，可信服务启动配置，见下表 |
| `benchmark_env` | 可选，固定测试环境变量；不可引用可调参数 |
| `search_space` | 必填，参数名到非空有限标量列表；标量为字符串、布尔值或有限数值 |
| `baseline` | 必填，键必须与参数域完全相同，每个值必须在域内 |
| `constraints` | 默认 `[]`，条件约束列表 |
| `scenarios` | 必填，至少一个命名场景 |
| `requirements` | 默认 `{}`，全局硬门槛 |
| `objective` | 必填，达标配置的排序目标 |
| `budget` | 可选，运行预算，见第 6 节 |
| `llm` | 必填，规划 API 配置，见第 5 节 |

| `server` 字段 | 含义 |
|---|---|
| `command` | 必填，非空 argv 字符串数组；不通过 shell 执行 |
| `env` | 默认 `{}`，环境变量名到字符串的覆盖值 |
| `health_url` | 必填，健康检查 URL |
| `startup_timeout_s` | 默认 `600`，正数 |
| `shutdown_timeout_s` | 默认 `20`，正数 |

| `scenarios[]` 字段 | 含义 |
|---|---|
| `name` | 必填且唯一，仅字母、数字、下划线、连字符和点 |
| `command`、`result_file` | 必填，固定评测 argv 与结果路径 |
| `metrics_format` | 默认 `canonical`；也支持 `vllm` |
| `timeout_s` | 默认 `600`，正数 |
| `requirements` | 默认 `{}`；同名指标覆盖全局门槛对象 |
| `before_each_command` | 可选，每次测量前的固定清理/准备 argv |
| `warmup_command` | 可选，每次测量前的固定预热 argv |
| `quality_command`、`quality_result_file` | 可选但必须成对出现 |
| `telemetry_file` | 可选，请求元数据 JSONL；声明后缺失或非法会使测量失败 |

每次重复按 `before_each → warmup → benchmark → quality` 执行，并使用新结果目录。统一冷/热缓存政策，防止前一轮结果被误读。遥测、性能、质量文件以及控制器生成的文件不能占用同一路径。

命令和环境字符串支持 `{param.NAME}`、`{env:NAME}`、`{repo_dir}`、`{trial_dir}`。布尔参数展开为 `true`/`false`；CLI 开关仍需按实际 vLLM 版本组织 argv。每个可调参数都必须绑定到 `server.command` 或 `server.env`。

`workflow` 禁止场景引用 `{param.*}`。未显式填写 `benchmark_env` 时，工作流冻结基线的 `server.env` 作为评测环境，避免 profile 影响测试代码。环境覆盖中的引用从原始继承环境展开，不按变量排列顺序相互展开。

条件约束和目标示例：

```json
{
  "constraints": [
    {"when": {"max_num_seqs": 128},
     "require": {"max_num_batched_tokens": [4096, 8192]}}
  ],
  "requirements": {"p99_ttft_ms": {"max": 2000}, "quality_score": {"min": 0.95}},
  "objective": {"metric": "output_tokens_per_second",
                "direction": "maximize", "scenario_aggregation": "min"}
}
```

`when` 中所有条件相等时，`require` 中每个参数都必须落入声明列表。约束只能引用已有参数与合法取值。门槛支持 `min`、`max` 或同时存在；二者均包含边界。

目标 `direction` 默认 `maximize`，也可 `minimize`；`scenario_aggregation` 默认 `min`，也可 `mean`/`max`。先取各场景重复测量的目标中位数，再跨场景聚合。最小化场景最坏延迟时通常选择 `minimize + max`。

### 指标适配与缺失值

`canonical` 接受指标名到数值的 JSON 对象。声明的目标或门槛指标必须存在且有限。错误计数必须是非负整数，不能把未知失败数当作零。

`vllm` 适配常见字段：`output_throughput → output_tokens_per_second`、`request_throughput → request_per_second`、`total_token_throughput → total_tokens_per_second`、`request_goodput → goodput_requests_per_second`；也识别受支持的 TTFT/TPOT/ITL/E2E 百分位格式。实际工具输出仍需用样本核对。

只有同时提供 `completed`、`failed` 且总数非零时，才能推导 `error_rate = failed / (completed + failed)`。显式错误率与计数不一致会被拒绝。一个重复的 P99 就是该重复的统计结果，程序不把多轮 P99 合成为全量请求 P99。

## 4. 策略与研究配置参考

| 工作流字段 | 含义 |
|---|---|
| `schema_version`、`name` | 必须为 `1`；名称必填 |
| `contract` | 相对当前工作流文件的契约路径，不接受绝对路径 |
| `budget` | 覆盖契约预算中的指定项；全部策略共享 |
| `research` | 研究连接，见下表 |
| `profiles` | 非空策略目录，首项必须保留契约基线 |
| `verification` | 最终验证与文件冻结，见第 6 节 |

| `profiles[]` 字段 | 含义 |
|---|---|
| `id`、`description` | 均必填；ID 为 1–64 字符的字母、数字、下划线或连字符 |
| `enabled` | 默认 `true` |
| `server`、`search_space`、`baseline`、`constraints` | 可选；每个字段整体替换契约对应字段，不递归合并 |
| `requires_quality` | 默认 `false`；设 `true` 后要求每个场景有质量 hook 和对应门槛 |
| `quality_metric` | 默认 `accuracy`；可改为客户使用的 `quality_score` 等 |
| `compatibility.status` | 必填 `verified` 或 `unverified` |
| `compatibility.evidence` | 字符串列表；`verified` 必须非空 |
| `compatibility.reason` | `unverified` 必须非空；`verified` 可为空 |

profile 无权改动验收场景、阈值、LLM API 或健康检查地址。需要改变模型精度语义的策略，应明确 `requires_quality: true` 并配置真实评测；程序无法从任意启动脚本自动识别所有精度风险。

| `research` 字段 | 默认与含义 |
|---|---|
| `transport` | `mcp`，或 `a2a` |
| `endpoint` | MCP 默认 `https://mcp.deepwiki.com/mcp`；A2A 默认 `http://127.0.0.1:8766/a2a` |
| `repository` | 固定为 `vllm-project/vllm-ascend` |
| `timeout_s`、`max_calls` | 默认 `60` 秒、`12` 次 |
| `on_error` | 默认 `stop`；可设 `continue` |

DeepWiki 提供的是 MCP 接口。推荐直接配置 `transport: "mcp"`；控制器初始化 MCP、发现工具后调用仓库问答。研究结果用于提供依据，不授予执行代码的权限。

需要 Agent-to-Agent 连接时，在另一个终端运行本机桥接：

```bash
python3 -m ascend_poc research-agent --port 8766 \
  --mcp-endpoint https://mcp.deepwiki.com/mcp --timeout 60
```

工作流改为 `{"transport":"a2a","endpoint":"http://127.0.0.1:8766/a2a"}`。桥接提供 `/.well-known/agent-card.json`，实现固定版本 A2A 0.3.0 的同步 JSON-RPC 研究子集，再调用 DeepWiki MCP；它不是完整 A2A 平台。仅监听回环地址，没有公网认证、多租户或分布式任务调度。

研究失败或次数耗尽且 `on_error: "stop"` 时停止继续搜索，仍尝试确认已测可行候选；`continue` 允许在缺少新研究证据时继续受限搜索。报告会保留失败原因，不能将后者解释为已有来源支持。

## 5. 大模型 API 与提案格式

使用 OpenAI-compatible Chat Completions：`base_url` 为 API 前缀，客户端追加 `/chat/completions`。这不等于任何服务商都支持所有可选参数。

```json
{
  "base_url": "https://YOUR_PROVIDER_HOST/v1",
  "model": "YOUR_PROVIDER_MODEL_ID",
  "api_key_env": "POC_LLM_API_KEY",
  "timeout_s": 60,
  "max_tokens": 1200,
  "temperature": 0.2,
  "json_mode": true,
  "max_retries": 2
}
```

`base_url`、`model` 必填；其余字段默认值如上。`temperature` 范围 0–2，可设 `null` 以不发送；`json_mode: false` 不发送 JSON response format；`max_retries` 为 0–10 的整数。按供应商实际接口调整，不猜测可用模型 ID。

LLM 返回 `choices[0].message.content` 中的 JSON 对象，工作流提案结构如下；参数必须覆盖所选 profile 的完整参数域：

```json
{
  "profile_id": "baseline",
  "parameters": {"max_num_seqs": 64, "max_num_batched_tokens": 4096,
                 "gpu_memory_utilization": 0.85},
  "hypothesis": "提高并发后检查吞吐收益和延迟门槛",
  "evidence_ids": []
}
```

模型只选择已启用、已验证 profile 与合法参数；有研究证据时至少引用一个已提供的证据 ID，无证据时才使用空列表。非法、重复或 API 失败的提案可以触发合法域内的回退搜索，事件记录会区分。运行完成不代表每轮都成功使用了 LLM。

密钥只放 `api_key_env` 指定的环境变量中；规划密钥会从测量子进程环境剥离，也不能通过模板注入子进程。推理服务需要鉴权时使用独立密钥变量。端点不能包含用户名、密码、查询参数或 fragment；使用 HTTPS，或本机回环 HTTP。

`.env.example` 仅说明变量，程序不会自动加载 `.env`；运行前需将密钥导出到当前进程环境。API 调用按服务商规则计费，客户端超时不代表已发请求在服务端取消，也不撤销可能产生的费用。

## 6. 预算、硬门槛与独立复测

| `budget` 字段 | 默认值与含义 |
|---|---|
| `max_trials` | `12`，探索试验上限，包含基线；最终确认另行计数 |
| `max_wall_time_s` | `7200`，整个实验墙钟预算 |
| `repetitions` | `3`，探索阶段每场景重复次数 |
| `final_repetitions` | `3`，最终确认每场景重复次数 |
| `finalists` | `2`，最多确认的候选数量 |
| `confirmation_reserve` | `false`，启用观测成本驱动的确认储备及进场预算预检 |
| `cost_safety_factor` | `1.25`，成本估计的安全系数，至少为 `1` |
| `estimated_trial_s` | 可选正数；尚无可用观测时，一次完整探索 trial 的粗估秒数，含启动和全部探索重复 |
| `early_stop` | `false`，仅探索阶段在已执行重复出现硬门槛失败后停止剩余测量 |

未启用成本储备时，75% 是探索截止线；探索提前结束，确认立即使用全部剩余时间，并非固定只能使用最后 25%。服务启动、研究/LLM 调用、预热和测试都消耗时间。停止进程的清理宽限可能使退出稍晚于截止时刻。

启用 `confirmation_reserve` 后，执行器用完成 trial 的成本观测估计剩余确认成本，区分固定启动与随重复次数增长的测试开销，并按 profile 取值。未知 profile 可参考保守回退估计或操作者的 `estimated_trial_s`；没有估计不等于耗时为零。开始完整 ABA/CI 确认组前，如果估计不能容纳，记录 `skipped_insufficient_budget`，不先执行一半再宣称成功。此预检是预算决策，不是硬件时限保证；实际慢启动、干扰或未知成本仍可能超时。

旧配置未声明这些字段时仍关闭。新版 `examples/ascend-workflow.json` 与 `examples/ascend-agent-workflow.json` 显式启用 `early_stop` 和 `confirmation_reserve`；RSI 继承所引用工作流的设置，不暗中覆盖用户选择。

在配置中填写 `budget.estimated_trial_s` 后，可以不启动模型先查看确认组的粗估成本：

```bash
python3 -m ascend_poc workflow-validate --config customer-workflow.json --estimate
```

未提供估值时，`estimated_s` 为 `null`，表示未知；`measurements` 仍给出确认测量次数。这不包含全部探索与外部 API 时间，也不代表完整任务一定能在预算内结束。

RSI 子实验只探索参照和一个候选，通用工作流可探索多个候选。`single` 需要一次确认，`aba` 需要三次，`aba_ci` 需要 `3 × ci_blocks` 次。完整 trial 的时长不能直接按重复次数倍增，因为模型加载通常只发生一次。先在目标机器记录加载、预热、测试及质量检查成本，再设置预算。

所有场景、每一次重复、所有硬门槛必须通过。均值或高吞吐不能抵消某一轮质量、延迟或错误率失败。探索通过只代表有资格进入最终确认；`best.json` 只导出确认通过的结果。

`early_stop` 检查已实际执行并归一化的重复，不把尚未执行的场景当作失败依据。发生硬失败后保留已有指标、`infeasible_early` 状态及 `aborted_after` 位置；未执行部分不补零，也不作为完整可行记录。最终确认与 ABA 的参照测量始终跑完整矩阵。`trial-execution.json` 保存状态、成本分解及早停位置，方便区分“已测失败”和“没测完”。

成本记录中的 `startup_s` 是启动至健康检查通过，`benchmark_s` 包含逐重复的缓存/预热钩子和性能命令，`quality_s` 是质量命令，`other_s` 是其余开销；`total_s` 是完整耗时。估计只外推完整 trial，不把早停的短耗时当作完整矩阵的廉价成本。

推荐正式工作流使用 ABA 验证：

```json
{
  "mode": "aba",
  "max_baseline_drift": 0.05,
  "min_relative_improvement": 0.02,
  "drift_multiplier": 2.0,
  "frozen_files": []
}
```

`mode` 默认 `single`；上述其他默认值如示例。`single` 只独立复测候选；`aba` 对每个入围候选执行基线 A → 候选 C → 基线 B，每组都跑完整场景矩阵和 `final_repetitions`。可选 `confirmation_profiles` 是允许最终确认的 profile ID 列表；省略时不额外限制。它不会启用未验证策略，也不会让未入围候选获得确认。RSI 子实验将其限定为 `candidate`，避免候选落选后给参照自身做确认。

逐场景以目标中位数计算 `ref=(A+B)/2`、`drift=abs(A-B)/ref`。最大化的改进率为 `C/ref-1`，最小化为 `ref/C-1`。任一场景漂移超限会使验证无效；最差场景的改进率必须超过 `max(min_relative_improvement, drift_multiplier × 最大漂移)`。

`accepted` 表示候选通过客户门槛且验证有效；`improved` 表示另有足够的改进证据。两者不同：达标不一定显著优于基线。基线可未达客户 SLO，但作为参照必须有完整有效测量。比率验证要求正目标值，非正目标应使用 `single`。

### 可选的独立多组置信区间验证

原 `aba` 的漂移门槛是保守启发式，不是已校准的置信水平。需要同时评估重复测量的不确定度时，可在新任务中预先启用 `aba_ci`：

```json
{
  "mode": "aba_ci",
  "max_baseline_drift": 0.05,
  "min_relative_improvement": 0.02,
  "drift_multiplier": 2.0,
  "ci_blocks": 5,
  "ci_confidence": 0.95,
  "ci_max_comparisons": 8,
  "frozen_files": []
}
```

| 字段 | 默认值与限制 |
|---|---|
| `ci_blocks` | `5`，预定完整 ABA 组数，整数 `3–100` |
| `ci_confidence` | `0.95`，严格位于 `0.5` 与 `1` 之间 |
| `ci_max_comparisons` | `8`，预定比较次数上限，整数 `1–10000`；不得小于工作流 `finalists` 或 Agent `max_experiments` |

每个 block 执行完整 A/C/B，三者各自重新启动服务；每次启动内的 `final_repetitions` 仍用于固定场景的中位数与逐重复硬门槛。统计样本量是 **block 数**，不是同一启动内的压测次数。不能复用同一 trial 冒充多个独立 block，也不能在看到结果后只保留有利组。

可从 `examples/ascend-workflow-ci.json` 和 `examples/inference-rsi-ci.json` 了解完整配置关系。它们仍是需替换模型、命令、兼容性证据和预算的模板，不是已经验证的昇腾默认参数。

每场景、每组以原 ABA 中位数构造方向一致的对数比率：最大化为 `log(C / mean(A,B))`，最小化取反。对跨 block 的这些比率计算 Student t 单侧下界，自由度为 `ci_blocks - 1`；将错误率按“场景数 × `ci_max_comparisons`”作 Bonferroni 分配。报告中的几何平均相对收益与置信下界属于此 log-ratio 估计，不能把它标成原客户目标的中位数、合并请求 P99 或原始基线的跨代收益。

每个 block 的完整性、硬门槛和漂移保护继续生效，跨 block 的参照中心变化也要受限。晋升既要求全部组有明确收益，也要求全部场景的置信下界超过保守改进门槛；不会因为引入 CI 就取消漂移阻断。结果保留逐组裁决及 `relative_improvement_lower_bound`，便于解释“达标但收益证据不足”。

这个区间依赖独立 block、近似正态的 block 对数效果等统计假设。重新启动服务不自动消除温度、竞争进程和时间趋势；零观测方差也不证明没有未观测噪声。`ci_max_comparisons` 只覆盖预定的本任务比较范围，不是任意多次重跑任务后的整体置信保证。应先做同配置独立启动对照、检查现场波动，再决定是否采用此模式；它不是默认替代协议。方法依据见 [统计参考](references.md#统计与测量优化依据)。

冻结关键文件时，填入实际 SHA-256：

```bash
sha256sum /opt/customer/run_performance.py /opt/customer/acceptance-data.jsonl
```

每项格式为 `{"path":"/absolute/path/to/file","sha256":"64位小写十六进制摘要"}`，示意摘要必须替换。相对路径以 `--workdir` 为基准。声明文件不论大小都流式完整哈希，在 trial 前后检查；变化会使整个实验失效并撤销获胜导出。只冻结列出文件名的 manifest 不等于冻结这些文件内容，`[]` 也不表示模型和数据已全部固定。

## 7. Agent 轨迹回放与遥测

[agent-trace.json](../examples/agent-trace.json) 提供分叉/汇合示例；[ascend-agent-replay.json](../examples/ascend-agent-replay.json) 是性能契约，[ascend-agent-workflow.json](../examples/ascend-agent-workflow.json) 将其接入研究优化循环。

轨迹是本项目的固定输入 DAG，不直接兼容所有外部 trace 格式。最小依赖示例：

```json
{
  "schema_version": 1,
  "requests": [
    {"id":"a", "agent_id":"task-1", "messages":[{"role":"user","content":"固定首轮输入"}], "max_tokens":64},
    {"id":"b", "after":["a"], "delay_s":0.1, "agent_id":"task-1",
     "messages":[{"role":"user","content":"固定下一轮输入，包含已记录的上下文"}], "max_tokens":64}
  ]
}
```

请求字段为 `id`、`messages`、`max_tokens`，可选 `after`、`delay_s`、`agent_id`、`branch_id`、`context_version`。ID 唯一、依赖无环。`after` 默认空列表；根节点的 `delay_s` 相对回放起点，子节点相对最后一个父请求实际完成时间。等待依赖不占并发槽；父节点失败后，后代计为失败且不发送 HTTP 请求。

```bash
python3 -m ascend_poc.replay \
  --trace examples/agent-trace.json \
  --base-url http://127.0.0.1:8000/v1 --model poc-model \
  --output runs/replay/metrics.json --telemetry-output runs/replay/telemetry.jsonl \
  --concurrency 8 --timeout-s 60 --total-timeout-s 300
```

此命令要求服务已运行；工作流场景中则由控制器启动服务。`--api-key-env` 可指定独立推理密钥。默认并发 8、单请求超时 60 秒、总超时 300 秒；只重放一次，不自动重试生成请求。

`messages` 是完整记录输入，前一轮新生成的输出不会自动拼入后续请求；不执行真实工具调用。`delay_s` 只记录外部工具/思考等待，不能重复计入模型等待。`max_tokens` 是上限，EOS 会改变实际输出；必须使用服务返回的实际 token usage，缺失或非法 usage 计为请求失败。

TTFT 和请求 E2E 从依赖就绪开始计时，包含客户端排队。TPOT 是内容流块首尾跨度除以实际输出 token 数减一，不是逐 token ITL；多 token 仅收到一个内容块时不可观测。存在此类成功请求时，canonical `p99_tpot_ms` 会缺失，从而无法通过 TPOT 门槛；诊断字段 `observed_p99_tpot_ms` 不应代替它。单 token 请求不具备 TPOT 资格，另见覆盖率和计数字段。

同一 `agent_id` 的请求组成任务：全部成功才计入 `task_completed`，否则计入 `task_failed`。任务 E2E 从首个成员就绪到最后成员完成，包含依赖和工具等待；部分请求缺少 ID 时查看 `task_observation_coverage`。这衡量性能完成情况，不证明任务答案正确。

回放完整结束，即使包含已计数的请求失败，也可退出 0；用结果的 `error_rate` 等门槛验收。传输/全局超时可退出 2，配置/文件错误退出 1。避免只看进程退出码判断客户是否通过。

### 请求遥测

```bash
python3 -m ascend_poc workload-analyze \
  --input runs/replay/telemetry.jsonl --output runs/replay/workload-summary.json
```

每行 JSON 必含 `request_id`、`started_at_s`、`completed_at_s`；成功记录需 `input_tokens`、`output_tokens`。`status` 默认 `ok`，失败记录可省略未知 token 数。可选 `cached_tokens`、`restored_tokens`、`accepted_tokens_per_step`、`agent_id`、`branch_id`、`context_version`、`event`；若声明 event，应为 `request_completed`。

必须满足 `restored_tokens ≤ cached_tokens ≤ input_tokens`。未观测缓存时省略输入字段，摘要用 null 与覆盖率表达未知，不补零。输入最多 8 MiB/100000 条；不接受任意提示词文本字段。工作流只向规划/研究通道发送受限数值摘要，不发送请求标识或原始 messages；原始输入仅用于目标推理服务。

统计按完成时间落入 60 秒窗口，保留空闲和尾部不完整窗口信息。`active_agent_count` 是观察到的不同成功 Agent 数，不是 decode 并发；上下文版本变化不证明发生了 KV 驱逐；attention proxy 也不是实测 prefill 时间。

## 8. P/D 容量预测：shadow 模式

```bash
python3 -m ascend_poc adaptive-plan \
  --config examples/adaptive-shadow.json \
  --windows examples/adaptive-windows.json --output runs/shadow.json
```

[adaptive-shadow.json](../examples/adaptive-shadow.json) 和 [adaptive-windows.json](../examples/adaptive-windows.json) 都是合成示例。命令是独立的纯预测器，不连接后端、不自动拟合参数、不迁移 KV、不改变 P/D 角色，也不验证客户 SLO。

| 配置字段 | 含义 |
|---|---|
| `schema_version`、`mode` | 必须为 `1`、`shadow` |
| `workers`、`current_prefill` | 同构 worker 总数 2–1024；当前 prefill 数至少 1，且至少保留 1 个 decode |
| `seconds_since_last_switch` | 可选非负秒数，用于冷却判断 |
| `calibration.source`、`hardware`、`model` | 必填非空校准来源、硬件、模型说明 |
| `calibration.prefill_coefficients_ms` | 必填 4 个非负系数：固定迭代、新 token、attention proxy、恢复 token 项 |
| `calibration.decode_coefficients_ms` | 必填 3 个非负系数：固定、batch、专家覆盖项 |
| `calibration.experts`、`experts_per_token` | 必填正整数，后者不大于前者 |
| `calibration.forward_tokens` | 必填且至少为 1 |
| `calibration.max_running_requests`、`kv_capacity_tokens` | 必填正整数，限制 batch 与近似 KV 容量 |
| `policy.min_windows` | 默认 `3`，可配整数且至少 2 |
| `policy.standard_errors`、`min_relative_gain` | 默认 `2`、`0.02` |
| `policy.max_observation_gap_s`、`cooldown_s` | 默认 `15`、`120` 秒 |

窗口 JSON 是列表，每项必须有 `start_s`、`end_s`、`completed_requests`、`mean_uncached_tokens`、`mean_cached_tokens`、`mean_output_tokens`、`mean_accepted_tokens_per_step`、`decode_population`、`mean_new_tokens_per_prefill_iteration`、`restore_fraction`、`max_observation_gap_s`；可选 `mean_causal_attention_pairs`。

窗口必须等宽、正时长、有序且不重叠；完成数为正整数，输出/接受率/prefill 迭代均值为正数，恢复比例在 0–1 内。最多 10000 个窗口，CLI 文件上限 8 MiB。缺观测、时间间隔过大或窗口不足时输出 `hold`；配置本身非法则报错。

系数必须在客户实际昇腾模型/并行策略上校准，不能直接沿用 GPU 数值。`decode_population`、恢复量等需要引擎/路由器采集，单靠 Chat Completions usage 或不同 Agent 数量无法获得，`workload-analyze` 输出也不能直接冒充完整容量窗口。

候选需要跨窗口持续收益，并超过标准误与相对收益阈值；冷却条件成立时保持现状。计算采用同构 worker、平均长度和近似 KV 容量约束，输出明确为 `prediction_only`、`calibration_verified: false`。程序不检查窗口相对当前时间的新鲜度，结果适合历史 shadow 分析。

## 9. 结果、恢复与交付

| 工作流产物 | 用途 |
|---|---|
| `workflow_report.md` | 汇总全部试验、门槛和最终确认 |
| `workflow.json`、`contract.json` | 本次配置快照 |
| `workflow_state.json`、`events.jsonl` | 恢复状态、带哈希链的事件与 checkpoint |
| `frozen-manifest.json` | 文件冻结清单 |
| `anytime.csv` | 随时间变化的独立确认最优值，不是探索最好值 |
| `research/` | 研究问答及相关证据 |
| `trials/` | 实际配置、分析、日志、原始指标与遥测摘要 |
| `trials/*/verification.json` | 最终确认的详细判定 |
| `trials/*/trial-execution.json` | 本轮执行状态、成本分解与可选早停位置 |
| `best.json`、`best-contract.json` | 仅确认通过时存在；获胜参数及完整固定配置 |

报告中的 `ACCEPTED` 为验收通过，`REJECTED` 为候选不达标，`INVALID` 为参照、完整性、噪声或执行条件不足。`infeasible_early` 是部分探索测量已失败，`skipped_insufficient_budget` 是确认未开始，二者都不冒充完整独立裁决。`status: completed` 只表示流程结束。没有 `best.json` 就没有独立确认的可交付配置；研究停止、预算不足和候选失败都应如实说明。

预算或确认策略导致的跳过记录在 `workflow_state.json` 的 `verification_attempts` 和事件链中，不生成虚假的通过 trial。CI 的 `reference_blocks` 保存各组 before/candidate/after 的 trial ID；前面的候选组使用 `confirmation_block`，最终候选确认记录承载整组裁决，单个中间 block 不能导出赢家。

```bash
python3 -m ascend_poc workflow --config customer-workflow.json \
  --output runs/customer-poc --workdir "$PWD" --resume
python3 -m ascend_poc report --output runs/customer-poc
```

恢复要求相同配置、工作目录、运行指纹、文件与有效事件链；沿用原始绝对截止时间，暂停时间也消耗预算。ABA 中断后不拼接不完整对照组，恢复时重新执行该组。修改测试、版本或数据后新开实验目录。

`report` 自动识别 workflow 或旧 tune 产物，根据保存快照重建 Markdown，不调用模型、研究服务或硬件。工作流会核对快照指纹、事件链和 checkpoint；`integrity_valid: false` 时拒绝整份重建，原始失败证据仍保留。旧 `tune` 仅校验契约指纹，没有相同的状态/事件链保护。此命令不重新验证今日环境，也不重新认证历史成绩；输出正被运行进程占用时拒绝改写。

`best-contract.json` 可用 `tune --planner random` 在新目录复测固定赢家，但不保留工作流的 ABA 分组。需要同样的 ABA 证据，应以原工作流启动新实验。正式交付同时保存配置、环境说明、原始测量、质量结果、确认结果，并在客户保留集复验。

运行指纹包括引用环境值的哈希、软件/Python/控制器信息和可识别直接命令文件；大文件的普通指纹可能仅记录元数据，关键内容应用 `frozen_files` 显式冻结。审计哈希链用于一致性检测，不提供针对同权限恶意修改者的隔离，也不等同于完整 Harbor 沙箱评测。

v0.6.0 只在同一次完整性检查内对多个 profile 共用的源码、解释器及命令文件去重；下一次检查重新读取，不把上一轮摘要当作内容未变的证明。冻结文件仍完整哈希，事件链与 checkpoint 的绑定保持不变。

密钥、客户提示词、日志、权重、私有测试和 `runs/` 不应提交 Git。`context`、兼容性证据、固定命令与本地结果也可能包含客户信息，发布前按实际内容检查。不要将密钥写入 JSON、命令参数或 URL。

## 10. CLI、排障与验证边界

下表命令均以 `python3 -m ascend_poc` 为前缀；安装后可换为 `ascend-poc`。

| 子命令 | 参数与用途 |
|---|---|
| `doctor` | 无参数；本机版本/设备信息 |
| `validate` | `--config FILE`；校验单契约 |
| `workflow-validate` | `--config FILE`；校验完整工作流与策略 |
| `workflow` | `--config FILE --output DIR [--workdir DIR] [--resume]`；推荐主入口 |
| `workflow-demo` | `[--output DIR]`，默认 `runs/workflow-demo`；CPU 全流程演示 |
| `research-agent` | `[--port 8766] [--mcp-endpoint URL] [--timeout 60]`；本机 A2A 桥接 |
| `workload-analyze` | `--input JSONL --output JSON`；请求遥测统计 |
| `adaptive-plan` | `--config JSON --windows JSON --output JSON`；P/D shadow 建议 |
| `report` | `--output DIR`；重建 workflow 或 tune 报告 |
| `agent` | `--config FILE --output DIR [--workdir DIR] [--resume] [--ignore-memory]`；持续实验 Agent，见 [RSI 指南](rsi.md) |
| `tune` | `--config FILE --output DIR [--planner llm\|random] [--workdir DIR] [--resume]`；兼容的单契约参数优化 |
| `demo` | `[--output DIR]`，默认 `runs/demo`；旧 tune CPU 演示 |

独立回放入口为 `python3 -m ascend_poc.replay`，参数见第 7 节。`tune` 没有工作流策略目录、DeepWiki/A2A 查询和 ABA 验证；它适合固定配置复测、参数域验证或兼容原有脚本，产物为 `state.json`、`report.md` 等。

`workflow`/`tune` 退出码：`0` 有确认赢家，`2` 没有确认可行方案，`1` 配置或运行异常，`130` 中断。其他命令和回放的退出码应结合各自输出解释。

| 现象 | 检查与处理 |
|---|---|
| `unbound parameters` / 非法约束 | 参数必须绑定服务 argv/env，基线和约束值都在有限域内 |
| 只有基线可选 | 检查其他 profile 的 `enabled`、`verified` 和证据；不要为增加候选虚标兼容 |
| 服务启动失败/OOM | 查 trial 服务日志、安装版本、卡占用、模型路径、并行规模和内存余量 |
| 端口或资源锁被占用 | 使用专用空闲端口；等待同资源作业结束，勿接管已有生产服务 |
| API 401/限流/JSON 错误 | 检查密钥变量和模型 ID；按供应商支持关闭 JSON mode/temperature；查看回退事件 |
| DeepWiki/A2A 失败 | 检查网络和本机桥接；需要继续无新证据搜索时显式选择 `on_error: continue` |
| 指标缺失/错误率不一致 | 查看原始 JSON 和 adapter；不要补虚假零值；核对计数分母 |
| TPOT 缺失 | 查看流块、真实 usage 和 TPOT 覆盖率；不可用观察子集代替全量门槛 |
| ABA 为 INVALID | 检查基线漂移、预算是否容纳完整组、文件或环境是否变化 |
| 确认因预算被跳过 | 检查观测耗时/粗估值、重复数、CI block 数和剩余时间；跳过不表示候选不达标 |
| `aba_ci` 达标但不晋升 | 查看逐组收益、置信下界、跨组漂移及预定比较次数；不能事后缩小组数或比较上限 |
| resume 拒绝或没有预算 | 保持原配置/指纹/日志；预算不因重启延长，变化后开新目录 |
| shadow 一直 hold | 检查完整等宽窗口、引擎采集、间隔、校准与冷却；不要降低门槛制造收益 |

开发或环境迁移后的主要检查：

```bash
python3 -m unittest discover -s tests -v
python3 -m ascend_poc workflow-validate --config examples/ascend-agent-workflow.json
python3 -m ascend_poc adaptive-plan --config examples/adaptive-shadow.json \
  --windows examples/adaptive-windows.json --output runs/check-shadow.json
python3 -m ascend_poc workflow-demo --output runs/check-workflow
python3 -m ascend_poc report --output runs/check-workflow
```

这些检查覆盖本地逻辑、HTTP 协议夹具和合成闭环，不等于真实 API 供应商兼容性、昇腾性能、模型质量或客户验收通过。真实结论必须来自固定客户契约下的硬件测量与独立确认。
