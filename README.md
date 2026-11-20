# Ascend PoC Autotuner

面向售前 PoC 的昇腾推理自动调优工具：**固定客户验收条件，测量并分析结果，查询 DeepWiki，调用大模型 API 选择已验证的推理策略与参数，最后独立复测。** 优先满足客户的质量、延迟和可靠性要求，再优化约定的性能指标。

Python 3.10+ · Linux · 无运行时第三方依赖 · v0.4.1

## 快速开始

在仓库根目录运行，无需 NPU、vLLM 或 API key：

```bash
python3 -m ascend_poc workflow-demo --output runs/quickstart
python3 -m ascend_poc report --output runs/quickstart
```

演示实际运行本地 HTTP API 替身、服务及测试子进程，走通 A2A → DeepWiki MCP 替身 → LLM 提案 → 测试 → ABA 复核 → 报告。数据是合成数据，不代表昇腾成绩。再次运行请使用新目录。

查看 `runs/quickstart/workflow_report.md`。只有最终复测通过，才会生成 `best.json` 和可复用的 `best-contract.json`；不会自动部署到生产。

## 文档

| 文档 | 内容 |
|---|---|
| **[完整使用指南](docs/guide.md)** | 部署、客户契约、LLM API、MCP/A2A、策略配置、Agent 回放、验收、恢复和排障 |
| [架构说明](docs/architecture.md) | 模块职责、数据流、执行边界和审计机制 |
| [参考来源](docs/references.md) | EvolvingInfra、ResearchBoost 与本项目的对应关系和适用范围 |
| [开发约定](AGENTS.md) | 为开发者和 coding agent 提供修改与验证规则 |

## 客户 PoC 工作流

1. 在专用昇腾环境手动跑通服务和客户评测，固定模型、权重、卡数、负载及缓存政策。
2. 从 [vllm-ascend.json](examples/vllm-ascend.json) 或 [customer-template.json](examples/customer-template.json) 建立验收契约；客户要求精度时接入真实质量评测。
3. 从 [ascend-workflow.json](examples/ascend-workflow.json) 建立策略目录，引用契约，并填写实际版本与兼容性证据。
4. 配置大模型 API 环境变量，检查配置，执行工作流，审阅最终报告。

```bash
python3 -m ascend_poc doctor
python3 -m ascend_poc workflow-validate --config customer-workflow.json
python3 -m ascend_poc workflow \
  --config customer-workflow.json --output runs/customer-poc --workdir .
```

`customer-workflow.json` 是按指南准备的本地文件。示例中的数值、设备、路径和命令需要替换；`workflow-validate` 只验证配置，不证明硬件兼容。模板仅启用基线，其余策略需在现场验证后启用。API key 从环境变量读取，不能写入配置或提交到仓库。

## 能力与边界

| 能力 | 当前实现 |
|---|---|
| LLM 调优 | OpenAI-compatible Chat Completions；在有限、已批准的策略和参数中选择 |
| 技术研究 | 直接调用 DeepWiki MCP，或经本项目的 A2A 0.3 同步研究代理查询 |
| 测试与验收 | 多场景、重复测量、质量钩子、硬门槛、最终 ABA 基线复核 |
| 证据与恢复 | 命令及原始指标、快照、文件指纹、事件哈希链、预算内断点恢复 |
| Agent 工作负载 | 固定请求依赖图回放、token/cache 元数据汇总和瓶颈分析 |
| 动态 P/D | 基于现场校准参数的影子建议；不执行在线角色切换 |

主要架构参考为 **EvolvingInfra**，评测方法参考 ResearchBoost。当前执行器通过重启服务测试配置，尚未移植定制 SGLang 引擎、跨节点 KV 管理或无重启 P/D 切换。所有候选都必须保持客户负载及验收规则，LLM 和研究文本不能修改评测器或执行任意命令。

本地自动化测试与合成端到端工作流已通过，直接 MCP 和 A2A → MCP 的真实 DeepWiki 查询曾验证成功。**尚未在真实昇腾卡或付费大模型 API 上验证性能。** 未观测的 token/cache 数据不会被当作实测零值；未通过最终复核的配置不会作为赢家导出。

## 开发验证

```bash
python3 -m unittest discover -s tests -v
python3 -m ascend_poc workflow-validate --config examples/ascend-workflow.json
python3 -m ascend_poc workflow-validate --config examples/ascend-agent-workflow.json
```

主入口为 `python3 -m ascend_poc workflow`。单方案搜索保留 `tune`；其余命令和配置字段统一见 [使用指南](docs/guide.md)。运行记录位于 `runs/`，默认不提交；源码在 `ascend_poc/`，测试在 `tests/`，模板在 `examples/`。
