# AgentNet-DA: 去中心化多智能体数据分析系统

基于 [AgentNet (NeurIPS 2025)](https://arxiv.org/abs/2504.00587) 的去中心化多智能体 DAG 架构，实现零训练的 NL2SQL 自然语言数据分析。4 个专业化 Agent（SQL 专家、数据分析师、可视化专家、报告撰写者）通过自适应 DAG 拓扑协作，将自然语言问题转化为 SQL 查询、数据分析和可视化报告。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY

# 3. 生成自建测试数据集
python data/build_dataset.py

# 4. 启动 Gradio Demo
python run.py demo

# 5. 运行评估
python run.py eval --rounds 1   # 快速验证（1轮）
python run.py eval              # 完整评估（3轮）
```

## 项目结构

```
project/
├── src/                     # 核心代码
│   ├── config.py            # Pydantic 配置管理
│   ├── llm_client.py        # DashScope API 客户端（OpenAI 兼容）
│   ├── datasource.py        # SQLite 数据源适配器
│   ├── nl2sql.py            # NL2SQL 引擎
│   ├── agents.py            # 4 Agent（Router + Executor）
│   ├── dag.py               # AgentNet DAG 引擎（进化 + 边权重）
│   ├── memory.py            # 经验池（RAG 检索 + FAISS）
│   ├── pipeline.py          # B1-B5 + 3 消融实验编排
│   └── evaluator.py         # 自动评估（EX/EM + 分析题评分）
├── app/
│   └── gradio_app.py        # Gradio Web UI
├── tests/                   # 67 个单元测试
├── data/                    # 数据集
│   ├── build_dataset.py     # 数据集生成脚本
│   ├── campus_trade.db      # 校园二手交易数据库
│   ├── food_delivery.db     # 外卖配送数据库
│   └── test_questions.json  # 自建 40 道测试题
├── run.py                   # 统一入口（demo / eval）
└── requirements.txt         # 6 个核心依赖
```

## 架构

```
用户输入自然语言问题
        │
        ▼
┌──────────────────────────────────────────┐
│           AgentNet DAG (4 Agents)         │
│                                           │
│  Agent-0: SQL 专家                        │
│    NL → SQL → 执行                        │
│         │ Forward                         │
│         ▼                                 │
│  Agent-1: 数据分析师                      │
│    趋势 + 异常 + 统计                     │
│         │ Forward                         │
│    ┌────┴────┐                            │
│    ▼         ▼                            │
│  Agent-2    Agent-3                       │
│  可视化     报告撰写                       │
│    │         │                            │
│    └────┬────┘                            │
│         ▼                                 │
│  边权重更新 + 能力进化 + 经验池存储       │
│  （纯数学运算，零训练）                   │
└──────────────────────────────────────────┘
        │
        ▼
  Gradio Web UI  ←→  DashScope API (Qwen3.5-9B)
```

## AgentNet 核心机制

| 机制 | 实现 | 说明 |
|------|------|------|
| **去中心化路由** | Router 自主决策 Execute/Forward/Split | 无需中央控制器 |
| **DAG 拓扑** | 完全有向图 + 边权重自适应 | 弱连接自动剪枝 |
| **能力进化** | ability += 0.1（成功）/ ability × 0.9（失败） | 零训练的渐进优化 |
| **经验池** | embedding + FAISS 检索历史成功轨迹 | 类 RAG 的 few-shot 增强 |

## Baseline 设计（8 种方法）

| # | 方法 | 说明 |
|---|------|------|
| B1 | Single Direct | 单模型直接生成 SQL |
| B2 | Single CoT | 单模型 + Chain-of-Thought |
| B3 | NL2SQL Only | 仅 Agent-0 独立工作 |
| B4 | 2-Agent Sequential | Agent-0 → Agent-3 两阶段 |
| **Ours** | **4-Agent DAG** | 完整 AgentNet 流水线 |
| 消融 | w/o Evolution | 去掉能力进化和边权重 |
| 消融 | w/o Experience | 去掉经验池 |
| 消融 | w/o Split | 仅允许 Execute + Forward |

## 评估

```bash
# 快速验证（1 轮，约 10 分钟）
python run.py eval --rounds 1 --source self-built

# 完整评估（200 题 × 8 方法 × 3 轮）
python run.py eval

# 只看某个数据集
python run.py eval --source spider
```

输出 7 张结果表到 `outputs/`：
- 总汇总表（均值 ± 标准差）
- 5 个数据源分表（自建 / Spider / BIRD / Spider2.0）
- 难度分层分析
- 消融贡献度分析

## 运行测试

```bash
pip install pytest
python -m pytest tests/ -v    # 67 tests
```

## 技术栈

| 层 | 技术 |
|----|------|
| LLM | Qwen3.5-9B (DashScope API / 本地 Ollama) |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 多 Agent | AgentNet DAG (NeurIPS 2025) |
| NL2SQL | 自研引擎（基于 Qwen3.5-9B） |
| 数据源 | SQLite（支持 Schema 自动提取） |
| 可视化 | Plotly |
| Web UI | Gradio 5 |
| 向量存储 | FAISS |
| 配置 | Pydantic Settings |

## 论文引用

本项目基于以下工作：

- **AgentNet**: Yang et al., "AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems", NeurIPS 2025.
- **Spider**: Yu et al., "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Database Semantic Parsing and Text-to-SQL Task", EMNLP 2018.
- **BIRD**: Li et al., "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs", NeurIPS 2023.

## 许可证

MIT
