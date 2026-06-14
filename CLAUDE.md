# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

**IMPORTANT**: All `project/` source files are deleted from the working tree (they are tracked in git). Restore them first:

```bash
git checkout HEAD -- project/
```

Then install dependencies and run:

```bash
cd project
pip install -r requirements.txt
python run.py train    # Train collaborative agents
python run.py eval     # Evaluate all 5 baselines
python run.py demo     # Launch Gradio demo on 0.0.0.0:7860
```

For China-based users, prefix with `HF_ENDPOINT=https://hf-mirror.com` or set it in your environment.

## Project Overview

Course project for **智能科学综合课程设计五** at **河北师范大学** (Hebei Normal University).
Instructor: 霍丽娜 (Huo Lina) — huolina@hebtu.edu.cn

**Core idea**: Two Qwen3-0.6B agents with role-specific LoRA + MAAC (Multi-Agent Actor-Critic) reinforcement learning for collaborative writing, compared against 5 baselines. Adapted from CoMLRL (github.com/OpenMLRL/CoMLRL).

## Common Commands

```bash
cd project
pip install -r requirements.txt

# Unified entry point (recommended)
python run.py train    # Train collaborative agents (defaults to Qwen3-0.6B)
python run.py eval     # Evaluate all 5 baselines (defaults to Qwen3-0.6B)
python run.py all      # Train + eval
python run.py demo     # Launch Gradio demo on 0.0.0.0:7860

# Direct script invocation (note: these CLI scripts default to Qwen2.5-0.5B, not 0.6B)
python src/train.py --model_name Qwen/Qwen3-0.6B --task tldr --dataset_size 320
python src/evaluate.py --model_name Qwen/Qwen3-0.6B --num_samples 50
python app/gradio_app.py --port 7860

# Kaggle: upload kaggle_notebook.ipynb to Kaggle with T4×2 GPU
```

**Model defaults**: `run.py` and `gradio_app.py` default to `Qwen/Qwen3-0.6B` (production). `train.py` and `evaluate.py` CLI scripts default to `Qwen/Qwen2.5-0.5B-Instruct` (lighter for development). Always pass `--model_name Qwen/Qwen3-0.6B` when invoking scripts directly.

**GPU requirement**: Training expects 2 GPUs by default (Agent A on GPU 0, Agent B on GPU 1). The code auto-falls back to a single GPU if only one is available, but this doubles VRAM usage (~8 GB total).

## Architecture

### Dual-Agent Setup

Two independently-trained Qwen3-0.6B agents with distinct roles:

| Agent | Role | Prompt Objective |
|-------|------|-----------------|
| Agent A | Concise summarizer / helper-function writer | ~220 characters, extract key points |
| Agent B | Detailed writer / main-function caller | 2-3× longer than A, expand with detail |

Agent B receives Agent A's output as `[Reference]` in its prompt. Generation is sequential: A generates first, then B generates using A's output as context.

### Model Wrapping ([actor_critic.py](project/src/actor_critic.py))

Each base model goes through three wrapping layers:

1. **QLoRA (4-bit)**: `BitsAndBytesConfig` with nf4 quantization and double quantization — enables training on 8GB GPUs
2. **LoRA adapters**: r=8, alpha=16, `lora_dropout=0.05`, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj` — separate adapter weights trained for Agent A and Agent B
3. **ValueHead** (`CausalLMWithValueHead`): a 2-layer MLP (`hidden_dim → Tanh → Linear(1)`) attached to the final hidden state. Predicts state value V(s) for the critic. Exposes `forward()` returning `ActorCriticOutput` (logits + values) and a `generate()` proxy.

### MAAC Training ([train.py](project/src/train.py))

Uses simple REINFORCE with a learned value baseline.

**Per-batch loop** (batch size B=4):
1. Batch-generate Agent A outputs (left-padded prompts, `temperature=0.6`, `top_p=0.9`, `do_sample=True`)
2. Batch-generate Agent B outputs (B's prompt includes A's output as `[Reference]`)
3. Compute joint reward for all B samples
4. Batched MAAC update — one forward pass per agent for all B samples:
   - `advantage = reward - V(s)` where V(s) is the mean value over completion tokens
   - `actor_loss = -Σ(log_prob(token)) × advantage`
   - `critic_loss = MSE(V(s), reward)`
   - `total_loss = (actor_loss + 0.6 × critic_loss) / V` where V is the number of valid (non-zero-reward) samples
5. Single optimizer step after backward across all B samples (manual gradient accumulation)

**Key hyperparameters**: lr=1e-4, epochs=3, dataset_size=320, batch_size=4, lora_r=8, lora_alpha=16, lora_dropout=0.05, max_new_tokens=256. Generation: temperature=0.6, top_p=0.9, do_sample=True.

**Inference mode** (used by baselines and evaluation): `temperature=0.1`, `do_sample=False` for deterministic output.

Agents A and B have independent LoRA weights, saved per epoch (`agent_a_epochN`, `agent_b_epochN`) and as finals (`agent_a_final`, `agent_b_final`). Left-padding is used during generation (`tokenizer.padding_side = "left"`), restored to `"right"` afterward. Uses PyTorch's built-in `sdpa` attention, compatible with both T4 (SM 7.5) and RTX 5060 (Blackwell).

### Task Modes and Reward Functions

| Task | Reward Function | Dataset |
|------|----------------|---------|
| `tldr` (default) | `length_ratio_reward` — B should be 2-3× longer than A; A ~220 chars. `reward = 0.5×ratio_score + 0.5×length_score` | `trl-lib/tldr` |
| `coding` | `execution_reward` — +0.5 for valid syntax (`ast.parse`), +0.5 for successful `exec`; requires B to call A's helper function | 8 inline questions |

### 5 Baselines

| # | Method | Description |
|---|--------|-------------|
| B1 | Single Model | One Qwen3-0.6B does the entire task alone |
| B2 | Parallel | Two models generate independently, no communication |
| B3 | Sequential | A→B pipeline using untrained base models (no LoRA) |
| B4 | Discussion | One-round: A writes → B comments → A revises |
| Ours | Collaborative | LoRA-trained agents with ValueHead-based MAAC |

## Project Files Quick Reference

| File | Purpose | Notes |
|------|---------|-------|
| [run.py](project/run.py) | Unified entry point (train/eval/all/demo) | Defaults to Qwen3-0.6B |
| [src/train.py](project/src/train.py) | QLoRA+MAAC training, all 5 baseline functions | CLI defaults to Qwen2.5-0.5B; baseline functions duplicated in evaluate.py |
| [src/evaluate.py](project/src/evaluate.py) | 5-baseline evaluation harness | CLI defaults to Qwen2.5-0.5B; **Ours baseline has a bug** (see Known Issues) |
| [src/actor_critic.py](project/src/actor_critic.py) | ValueHead + CausalLMWithValueHead wrapper | Stable, no known issues |
| [src/utils.py](project/src/utils.py) | Seed, device info, result formatting | `format_results_table()` is VLM-specific — do not use for this project |
| [src/__init__.py](project/src/__init__.py) | Package init | **Wrong docstring** (says "Food Classification") |
| [app/gradio_app.py](project/app/gradio_app.py) | Web demo on port 7860 | Correctly loads trained LoRA adapters — reference for `PeftModel.from_pretrained()` usage |
| [requirements.txt](project/requirements.txt) | Dependencies | Missing `trl`; has unnecessary VLM deps (torchvision, scikit-learn, etc.) |

## Known Issues

### evaluate.py: "Ours" baseline loads fresh untrained model

In `run_evaluation()` (evaluate.py), when trained LoRA adapters exist at `lora_path/agent_a_final`, the code calls `load_agent_with_lora()` which creates **new random LoRA adapters** on a fresh base model — it never loads the saved adapter weights via `PeftModel.from_pretrained()`. Additionally, `agent_b` is never loaded even when `agent_b_path` exists (the code only checks `agent_a_path`).

**Impact**: The "Ours" baseline uses random-initialized LoRA weights, making it equivalent to (or worse than) B3 (Sequential).

**Fix reference**: [gradio_app.py](project/app/gradio_app.py) correctly loads trained adapters with `PeftModel.from_pretrained(base_model, adapter_path)` then `merge_and_unload()`. Follow that pattern to fix evaluate.py.

### Duplicated baseline functions

`single_agent_inference`, `parallel_inference`, `sequential_inference`, and `discussion_inference` are defined identically in both [train.py](project/src/train.py) and [evaluate.py](project/src/evaluate.py). evaluate.py imports from train.py but then redefines these functions locally. Changes to one copy will not affect the other — if you modify a baseline, check both files.

### src/__init__.py has wrong docstring

Currently reads `"""Qwen3-VL LoRA Fine-Grained Food Classification."""` — this is a leftover from a separate VLM project. The directory contains collaborative writing code.

### utils.py format_results_table() is VLM-specific

`format_results_table()` expects VLM classification metrics (Top-1 Acc, Top-5 Acc, description quality, inference speed in img/s, VRAM in GB). None of these fields are produced by the collaborative writing pipeline. Do not use this function. The evaluate.py script prints its own summary.

### Missing dependencies

`trl` (used by `train_sft()` for SFT warmup in train.py) and `wandb` (optional logging, with try/except guard) are not listed in `requirements.txt`. Install them separately if needed:

```bash
pip install trl>=0.12.0 wandb
```

Conversely, `requirements.txt` includes VLM-specific packages (`torchvision`, `scikit-learn`, `scipy`, `pillow`, `qwen-vl-utils`) only needed by the separate `prepare.py` VLM project — they can be skipped for the collaborative writing pipeline.

## extra/AgentNet-Code/ — Separate Sub-Project

`extra/AgentNet-Code/` (37 files, gitignored) contains the NeurIPS 2025 AgentNet supplementary code — a **separate multi-agent system** for BigBenchHard tasks using a DAG topology of GPT-4o-mini-powered agents with dynamic routing, ability evolution, and experience pools. It has its own CLAUDE.md at `extra/AgentNet-Code/CLAUDE.md`.

Quick start:

```bash
cd extra/AgentNet-Code
pip install -r requirements.txt
python run_agentnet.py --experiment_name bigbenchhard_new_abilities

# Single-agent GPT-4o-mini baseline
python run_baseline.py
```

Requirements: OpenAI API key (set in `config/setting.py`), `./big_datasets/bigbenchhard/` dataset directory (not included).

Notable pitfalls in AgentNet-Code:
- `single_task_baseline.py` line 30: syntax error (`save_path =` with no value)
- `src/utils.py` imports `HTTP_PROXY`/`HTTPS_PROXY` that don't exist in `config/setting.py`
- `models/` directory is legacy code, never imported by the main pipeline — ignore it

## Hardware Requirements

| Scenario | GPU | VRAM per GPU | Notes |
|----------|-----|-------------|-------|
| Training (QLoRA) | 2× T4 or RTX 5060 8GB | ~4 GB | Kaggle T4×2 is the primary training target; each agent on one GPU |
| Single-GPU training | 1× T4 16GB or better | ~8 GB | Both agents on one GPU via `device_map={"": 0}` |
| Inference (4-bit) | Any CUDA GPU | ~2 GB total | Two 0.6B models at 4-bit fit in 4 GB |
| Gradio demo | RTX 5060 8GB | ~4 GB | Loads base model + 2 merged LoRA adapters |

## Important Notes

- `project/prepare.py` is for a **separate VLM/Food101 project** (not the collaborative writing pipeline). It downloads Qwen3-VL models and Food101. Do not modify it for the main project.
- **No tests, no CI/CD, no build system** — this is a course project. Evaluation is manual via `run.py eval`.
- **Models saved as LoRA adapters only** (not full weights). Load with `PeftModel.from_pretrained(base_model, lora_path)` or via `load_agent_with_lora()`.
- For China-based users: set `HF_ENDPOINT=https://hf-mirror.com` to use the HuggingFace mirror.
- The Gradio demo supports `--share` for temporary public URLs.
