# Ascend Inference RSI Agent

**用自然语言写下客户在意的成绩，让 Agent 结合 vLLM-Ascend Wiki、目标机器支持的参数和实测反馈，自主寻找更好的推理配置。**

例如：“两张昇腾 A2，Qwen 模型，P99 首字不超过 2 秒，尽量提高输出吞吐。”设备和首字门槛来自这段需求；没写精度、缓存、批大小或并行布局，就不从旧模板替客户增加限制。

Python 3.10+ · Linux · 无运行时第三方依赖 · v0.7.0

## 新入口：说需求，跑实验，看证据

先用不需要昇腾卡或外部 API 的演示了解流程：

```bash
python3 -m ascend_poc autotune-demo --output runs/my-open-demo
python3 -m ascend_poc autotune-report --output runs/my-open-demo
```

打开 `runs/my-open-demo/report.md` 阅读过程与结论。演示中的规划大模型、Wiki 和性能数字使用本地替身，只证明流程可运行。再次演示使用新目录。

常规吞吐／延迟测试推荐 [现成 vLLM benchmark 模板](examples/open-vllm-bench.json)，不用另写性能脚本；设置真实的 `MODEL_PATH` 和 `BENCHMARK_TOKENIZER_PATH` 后使用。客户已有专用跑分程序时，使用 [自定义适配器模板](examples/open-runtime.json)。两者都需按现场设备、模型和 API 修改，详见 [准备真实环境](docs/open-tuning.md#5-准备真实环境)。

准备好 `customer-runtime.json` 后，先做纯结构校验，再输入需求：

```bash
python3 -m ascend_poc autotune-validate --runtime customer-runtime.json
python3 -m ascend_poc autotune \
  --runtime customer-runtime.json \
  --request '两张昇腾 A2，使用 Qwen 模型，P99 TTFT 不超过 2000 ms，最大化输出 tokens/s。' \
  --output runs/customer-open-001 --workdir "$PWD"
```

也可以改用 `--request-file customer-request.txt`。模型、设备名称应与运行接线中登记的实际资源一致；首次使用请跟随 **[自然语言全参数调优指南](docs/open-tuning.md)**，不要把占位模型路径当成可运行环境。

## 这次改变了什么

| 你关心的能力 | 新的 `autotune` 模式 |
|---|---|
| 不只搜索三个参数 | 从目标环境自己的 CLI 帮助发现推理选项，不再要求预列三个参数的候选值 |
| 没写的条件可以调整 | 只有需求中明确表达的条件进入客户限制；运行模板的起始值不自动成为锁 |
| 自动调整推理策略 | 支持布尔开关和嵌套 JSON，如 `additional-config`、`compilation-config`、`speculative-config`；可运行性由实际启动和测试验证 |
| 失败后知道怎样改 | 保留测量、约束违规和启动失败反馈，先分析，再查询 DeepWiki MCP，提出下一轮假设 |
| 不用先手工批准每个 profile | 新模式不依赖旧模式的 `verified` 策略目录；候选先通过结构、明确限制和资源检查，再实测 |
| 客户只给跑分目标 | 使用已接入的实际指标作为主目标；没有明说的质量／延迟门槛不会从旧模板继承 |
| 能解释为什么胜出 | 保存原文到正式规则的对应、候选配置、研究依据、测量和独立复核结果 |

“全参数”指**目标引擎实际提供、且可以通过当前运行接线执行的推理配置**。Agent 仍然不能变出不存在的 NPU、模型权重或分布式部署，也不能通过改评分脚本或伪造测量提高成绩。连接地址、凭据、结果路径和执行程序属于运行接线。详细边界见 [哪些能调](docs/open-tuning.md#3-所有参数具体包括什么)。

## 把它想成实验负责人

规划大模型是实验负责人：阅读需求和成绩、查资料、提出下一步。昇腾上的被测模型是参赛者：处理测试请求，产生实际性能指标。两者的模型名称和 API 角色不同。

一个回合包括：**观察成绩 → 分析瓶颈 → 查询 Wiki → 提出新配置 → 启动与测试 → 独立复核 → 保留结果或继续调整**。开启更多开关不等于更快；Agent 需要用结果检验自己的判断。起始配置无法测量或违反明确客户门槛时，首个可行候选经过三次独立启动、每次全部重复均达标的复核后，可建立 `bootstrap` 合格基线；不因原不合格方案分数更高而阻止它，也不宣称相对不合格／失败方案的加速比。

如果允许改变并发、上下文或模型版本，报告必须同时列出这些条件。不同条件下的较高分数可以是更好的工作点，但不能省略条件后宣称“相同负载下加速”。未设置或未执行质量测试时，报告也不能宣称答案质量已经通过。

这里的 **RSI（递归改进）** 是把实测结果与更好的配置反馈给下一轮；它不训练规划模型权重，不编写新 kernel，也不自动热切换生产服务。实验会逐配置启动和停止服务。

## 文档怎么读

| 文档 | 适合谁 |
|---|---|
| **[自然语言全参数调优](docs/open-tuning.md)** | 新模式的需求写法、运行接线、执行与结果解释 |
| [参数与策略通俗图解](docs/tuning-explained.md) | 想看懂各种参数、缓存、量化、并行的售前与非技术读者 |
| [从零复现](docs/quickstart.md) | 下载代码、选择演示路径，以及旧版三文件接入流程 |
| [受控模式基础手册](docs/guide.md) | 原 `workflow/tune` 的固定契约与配置字段 |
| [受控 RSI Agent 指南](docs/rsi.md) | 原 `agent` 的 profile、记忆、预算与严格恢复 |
| [架构说明](docs/architecture.md) / [参考来源](docs/references.md) | 模块职责、EvolvingInfra 与 ResearchBoost 的方法来源 |
| [开发约定](AGENTS.md) | 开发者与 coding agent 的修改规则 |

新开放模式当前使用本次任务内反馈和 ABA，不包含旧 `agent` 的跨任务经验库、`--ignore-memory` 或多组 CI。

`autotune` 是新的推荐入口；`agent`、`workflow`、`tune` 保留原行为，方便复现既有任务。旧的有限参数域和预验证 profile 不会暗中变成开放搜索。

```bash
python3 -m unittest discover -s tests -v
```

源码在 `ascend_poc/`，模板在 `examples/`。真实昇腾性能、客户质量与外部 API 兼容性需要在目标环境测量；客户配置、密钥、业务数据、日志与 `runs/` 不应提交到 Git。
