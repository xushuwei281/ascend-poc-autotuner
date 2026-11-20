# Ascend Inference RSI Agent

面向售前 PoC 的昇腾推理优化 Agent。**固定客户验收条件，让大模型自主选择研究、实验、反思和晋升步骤；只有实测达标且优于当前配置的候选，才能成为下一轮优化的起点。** 实验经验跨任务保存，帮助后续实验选择。

Python 3.10+ · Linux · 无运行时第三方依赖 · v0.5.0

本项目保留包名 `ascend-poc-autotuner`。这里的 RSI 是**配置与策略层的递归改进**：优化后的配置反馈给下一轮，实验经验反馈给 Agent。它不训练大模型权重、不自行改写 kernel，也不执行在线 P/D 角色切换。

## 快速开始

在仓库根目录运行，无需 NPU、vLLM 或外部 API key：

```bash
python3 -m ascend_poc agent-demo --output runs/rsi-demo
python3 -m ascend_poc agent-report --output runs/rsi-demo
```

演示使用本地协议替身与合成指标，展示工具决策、实验反馈和独立验证。查看 `runs/rsi-demo/agent_report.md`；只有成功晋升后，才会生成 `champion.json` 和 `champion-contract.json`。再次演示使用新目录，数字不代表昇腾实测成绩。

## 文档

| 文档 | 内容 |
|---|---|
| **[Inference RSI Agent 使用指南](docs/rsi.md)** | Agent 配置、工具循环、递归晋升、经验、预算、恢复与结果解读 |
| [基础测试与部署手册](docs/guide.md) | 昇腾部署前提、客户契约、API、MCP/A2A、策略、回放与评测接入 |
| [架构说明](docs/architecture.md) | Agent、实验执行器、独立裁决和影子预测之间的边界 |
| [参考来源](docs/references.md) | EvolvingInfra、ResearchBoost main 与 RSI-Infra Bench 的方法对应 |
| [开发约定](AGENTS.md) | 开发者与 coding agent 的修改和验证规则 |

## 用于客户 PoC

先按基础手册跑通客户服务及性能/质量评测，固定模型、卡数、负载、缓存政策和验收线。随后建立策略目录与 Agent 配置：

```bash
python3 -m ascend_poc agent-validate --config examples/inference-rsi.json
python3 -m ascend_poc agent \
  --config customer-agent.json --output runs/customer-rsi --workdir .
```

`customer-agent.json` 按 [Agent 指南](docs/rsi.md) 准备，引用客户工作流；API 从其契约读取。模板参数和路径需要替换，兼容性必须在现场验证。配置检查通过不代表硬件兼容或客户验收通过。密钥只从环境变量读取。

## 工作方式

| 环节 | 实际行为 |
|---|---|
| 观察与研究 | 查看状态、检索同环境经验、通过 DeepWiki MCP 或本地 A2A 桥查询资料 |
| 实验与反思 | 提出合法策略/参数及假设，运行固定评测，记录成功、失败和经验 |
| 独立晋升 | 候选通过客户门槛、ABA 复核及提升阈值，且参照仍是当前代，才可晋升 |
| 递归改进 | 新冠军成为下一实验参照；经验只帮助选候选，不继承历史成绩 |
| 结果交付 | 保存事件链、配置快照、实验记录与冠军配置；不自动部署生产 |

架构主要参考 EvolvingInfra 的负载感知，Agent 过程参考 ResearchBoost `main`，评测参考其 `bench`。执行器通过启动/停止服务测量配置；`adaptive-plan` 仍只给出 P/D 影子建议。**真实昇腾性能、客户质量和外部模型 API 兼容性需在目标环境验证。**

## 开发与兼容

```bash
python3 -m unittest discover -s tests -v
python3 -m ascend_poc agent-validate --config examples/inference-rsi.json
```

`workflow` 保留一次固定目录内的研究调优，`tune` 保留单方案参数搜索；原配置与命令继续使用，见 [基础手册](docs/guide.md)。源码在 `ascend_poc/`，模板在 `examples/`，测试在 `tests/`；运行日志、经验文件及客户配置不应提交到 Git。
