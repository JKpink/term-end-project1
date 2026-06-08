# 多智能体 LLM 协作写作

## 项目简介

探索小模型的能力边界：两个 Qwen3-0.6B 通过角色分工协作，能否超越单个大模型？

- **Agent A**: 精炼摘要者（提取核心要点，~220字符）
- **Agent B**: 详细写作者（基于 A 的摘要展开，2-3倍长度）

## 环境要求

- Python 3.10+
- Kaggle T4×2（训练）/ RTX 5060 8GB（推理）
- Windows / Linux

## 快速开始

```bash
# 安装依赖
cd project
pip install -r requirements.txt

# 本地训练（小规模测试）
python src/train.py --model_name Qwen/Qwen3-0.6B --dataset_size 50 --num_epochs 1

# 本地评测（5 baseline 对比）
python src/evaluate.py --num_samples 20

# Kaggle 训练（完整版）
# 上传 kaggle_notebook.ipynb，Run All

# Gradio Demo
python app/gradio_app.py
```

## Baseline 对比

| # | 方法 | 说明 |
|---|------|------|
| B1 | Single Model | 单模型完成全部任务 |
| B2 | Parallel Generation | 两模型独立生成，不通信 |
| B3 | Sequential Pipeline | A→B 顺序，不训练 |
| B4 | One-Round Discussion | A 和 B 各说一句话 |
| **Ours** | **Collaborative** | LoRA 训练角色分工 |

## 项目结构

```
project/
├── README.md
├── requirements.txt
├── kaggle_notebook.ipynb     # Kaggle T4×2 训练
├── run.py                    # 本地实验编排
├── src/
│   ├── train.py              # QLoRA + 双Agent协作训练
│   ├── evaluate.py           # 5 baseline 评测
│   └── utils.py              # 工具函数
└── app/
    └── gradio_app.py         # Web Demo
```

## 训练参数

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B (4-bit nf4) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| 学习率 | 1e-4 |
| Epochs | 3 |
| 训练数据 | 320 条 (TLDR) |
| 显存峰值 | ~4 GB |

## 参考

- CoMLRL: [github.com/OpenMLRL/CoMLRL](https://github.com/OpenMLRL/CoMLRL)
- 河北师范大学 智能科学综合课程设计五
