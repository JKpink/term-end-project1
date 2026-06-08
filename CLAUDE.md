# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Repository Overview

Course project for **智能科学综合课程设计五** at **河北师范大学** (Hebei Normal University).

## Project: Multi-Agent LLM Collaborative Writing

Train two small LLM agents with role specialization for collaborative writing.
Adapted from CoMLRL (github.com/OpenMLRL/CoMLRL).

### Core idea

**2 × Qwen3-0.6B + LoRA (collaborative) vs single Qwen3-1.7B**

- Agent A: concise summarizer (extract key points)
- Agent B: detailed writer (expand into full summary)

### Baselines

| # | Method | Description |
|---|--------|-------------|
| B1 | Single Model | One model does everything |
| B2 | Parallel | Two models, no communication |
| B3 | Sequential | A→B, no training |
| B4 | Discussion | One-round discussion |
| Ours | Collaborative | LoRA-trained role specialization |

### Key source files

| File | Role |
|------|------|
| `project/run.py` | Unified entry point: train, eval, all, demo |
| `project/src/train.py` | QLoRA + MAAC training, reward functions, formatters, baseline inference |
| `project/src/evaluate.py` | 5-baseline comparison, metrics |
| `project/src/actor_critic.py` | `CausalLMWithValueHead` + `ValueHead` — custom actor-critic architecture |
| `project/src/utils.py` | `set_seed`, `get_device_info`, result formatting |
| `project/app/gradio_app.py` | Web demo: 5-column comparison |
| `project/kaggle_notebook.ipynb` | Kaggle T4×2 training notebook |

### Common commands

```bash
cd project
pip install -r requirements.txt

# Unified entry point (recommended)
python run.py train    # Train collaborative agents
python run.py eval     # Evaluate all 5 baselines
python run.py all      # Train + eval
python run.py demo     # Launch Gradio demo

# Or use individual scripts
python src/train.py --model_name Qwen/Qwen3-0.6B --task tldr --dataset_size 320
python src/evaluate.py --model_name Qwen/Qwen3-0.6B --num_samples 50

# Kaggle: upload kaggle_notebook.ipynb for T4×2 training

# Gradio demo
python app/gradio_app.py
```

### Architecture: MAAC (Multi-Agent Actor-Critic)

The training wraps each base Qwen model in a custom actor-critic architecture:

1. **`load_agent_with_lora()`** ([train.py:203](project/src/train.py#L203)):
   - Loads base model with 4-bit QLoRA (`BitsAndBytesConfig`, nf4, double quantization)
   - Applies LoRA adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`) via PEFT
   - Wraps in **`CausalLMWithValueHead`** ([actor_critic.py:60](project/src/actor_critic.py#L60)) which adds a `ValueHead` — a 2-layer MLP (`hidden → Tanh → Linear(1)`) for critic value prediction

2. **`_maac_update()`** ([train.py:480](project/src/train.py#L480)):
   - **Actor loss**: `-log_prob(completion|prompt) × advantage` where `advantage = reward - V(s)`
   - **Critic loss**: `MSE(value, reward)`
   - **Combined loss**: `actor_loss + 0.6 × critic_loss`
   - Uses REINFORCE-style single reward signal (not full MAGRPO)

3. **Training loop** in `train_collaborative()`:
   - Pre-format all prompts (Agent A/B formatters)
   - Per batch: generate A outputs → generate B outputs (receives A's output as `[Reference]`) → compute reward → MAAC update per sample → optimizer step
   - Save LoRA adapters per epoch

### Task modes

| Task | Reward function | Dataset |
|------|----------------|---------|
| `tldr` (default) | `length_ratio_reward` — B should be 2-3× longer than A, A ~220 chars | `trl-lib/tldr` |
| `coding` | `execution_reward` — A writes helper function, B uses it; +0.5 syntax valid, +0.5 runs | Inline coding questions |

### Key design decisions

- **QLoRA (4-bit)**: Enables training on 8GB consumer GPU
- **Simple REINFORCE**: Not full MAGRPO — uses single reward signal per step
- **Length ratio reward**: Agent B should produce 2-3x longer output than Agent A
- **320 training samples**: Matches CoMLRL's TLDR example
- **Two separate LoRA adapters**: Agent A and Agent B have different LoRA weights
- **Default model**: `Qwen/Qwen2.5-0.5B-Instruct` in code, but `Qwen/Qwen3-0.6B` recommended for actual training
- **Batch training**: Batch size 4, gradient accumulation via manual optimizer stepping across batched MAAC updates

### Hardware

- Training: Kaggle T4×2 (16GB each) or RTX 5060 8GB
- Inference: 2×Qwen3-0.6B 4-bit ≈ 2GB

### Note

`project/prepare.py` is for a separate VLM/Food101 project (see `project/REPORT.md`) — not used by the collaborative writing project.

## Instructor

霍丽娜 (Huo Lina) — huolina@hebtu.edu.cn
