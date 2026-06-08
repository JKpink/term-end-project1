"""
Collaborative dual-agent training with QLoRA for dual-agent writing.

Adapted from CoMLRL's tldr-len-ratio.py.
Key additions: QLoRA (4-bit), LoRA fine-tuning, RTX 5060 / Kaggle T4×2 support.
"""

import os
import math
import argparse
from functools import partial
from typing import List, Dict, Any, Optional

import torch
try:
    import wandb
except ImportError:
    wandb = None
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
try:
    from trl import SFTTrainer
except ImportError:
    SFTTrainer = None
from actor_critic import CausalLMWithValueHead


# ─── Reward Functions ─────────────────────────────────────────────

def length_ratio_reward(
    completions_a: List[str],
    completions_b: List[str],
    ratio_min: float = 2.0,
    ratio_max: float = 3.0,
    target_short_chars: int = 220,
) -> List[float]:
    """Reward Agent A (concise) + Agent B (detailed) for correct length ratio.

    Agent A should produce ~target_short_chars characters.
    Agent B should be 2-3x longer than Agent A.
    """
    rewards = []
    for a, b in zip(completions_a, completions_b):
        len_a = len(a.strip())
        len_b = len(b.strip())

        if len_a == 0 or len_b == 0:
            rewards.append(-1.0)
            continue

        # Ratio score: B should be 2-3x A
        ratio = len_b / max(len_a, 1)
        if ratio_min <= ratio <= ratio_max:
            ratio_score = 1.0
        elif ratio < ratio_min:
            ratio_score = 1.0 - (ratio_min - ratio) / ratio_min
        else:
            ratio_score = 1.0 - (ratio - ratio_max) / ratio_max
        ratio_score = max(-1.0, ratio_score)

        # Length score: A should be ~target
        scale = target_short_chars / 2
        short_score = 1.0 - abs(len_a - target_short_chars) / scale
        short_score = max(-1.0, min(short_score, 1.0))

        combined = 0.5 * ratio_score + 0.5 * short_score
        rewards.append(float(max(-1.0, min(combined, 1.0))))

    return rewards


def execution_reward(
    completions_a: List[str],
    completions_b: List[str],
) -> List[float]:
    """Reward for coding: Agent A writes helper, Agent B uses it.

    +0.5 syntax valid, +0.1 per correct test case.
    Adapted from CoMLRL's leetcode-func-print.py.
    """
    import re
    import ast
    import io
    import contextlib

    rewards = []
    for c1, c2 in zip(completions_a, completions_b):
        reward = 0.0
        code_a = _clean_code(c1)
        code_b = _clean_code(c2)

        # Find function in A
        func_match = re.search(r"def\s+(\w+)\s*\(", code_a)
        if not func_match:
            rewards.append(0.0)
            continue

        func_name = func_match.group(1)
        # Check B uses A's function
        if func_name not in code_b:
            rewards.append(0.0)
            continue

        combined = f"{code_a}\n\n{code_b}"
        try:
            ast.parse(combined)
            reward += 0.5  # syntax valid

            # Try executing
            local_vars = {}
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(combined, local_vars)
            reward += 0.5  # runs without error
        except Exception:
            pass

        rewards.append(reward)
    return rewards


def _clean_code(code: str) -> str:
    """Remove markdown and explanatory text from generated code."""
    import re
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        # Keep code lines, skip explanatory text
        if stripped and not re.match(
            r"^(Here|This|The|Now|Let|We|In|Note|Make|You|I|Please|Remember|Below|Above)",
            stripped,
        ):
            lines.append(line)
    return "\n".join(lines).strip()


# ─── Prompt Formatters ────────────────────────────────────────────

def build_tldr_formatters(tokenizer):
    """Build dual-agent formatters for TLDR summarization."""
    def _format(system_prompt: str, example: dict) -> str:
        prompt = example.get("prompt", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    concise = "You summarize Reddit posts into a single concise TL;DR (about 220 characters)."
    detailed = (
        "You write detailed TL;DR summaries about 2-3x longer than a standard version. "
        "Build upon the concise summary provided by your collaborator."
    )
    return [
        partial(_format, concise),
        partial(_format, detailed),
    ]


def build_code_formatters(tokenizer):
    """Build dual-agent formatters for code generation."""
    def _format_a(example: dict) -> str:
        question = example.get("question", example.get("prompt", ""))
        prompt = (
            "Write a single self-contained Python helper function based on this requirement.\n"
            "Only output the function code, no explanations.\n\n"
            f"Requirement: {question}"
        )
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _format_b(example: dict) -> str:
        question = example.get("question", example.get("prompt", ""))
        prompt = (
            "Write a main function that uses a helper function (already defined) "
            "to solve this task. Include test examples with print statements.\n\n"
            f"Task: {question}\n\n"
            "Assume the helper function is already available. "
            "Only output the main function code and test cases."
        )
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    return [_format_a, _format_b]


# ─── Model Loading with QLoRA ─────────────────────────────────────

def load_agent_with_lora(
    model_name: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    use_4bit: bool = True,
    device_map: str = "auto",
):
    """Load a model with QLoRA for memory-efficient training."""
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        attn_implementation="sdpa",  # sdpa = PyTorch内置加速，兼容T4/5060
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Add LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Wrap with ValueHead for actor-critic
    model = CausalLMWithValueHead(model, attach_value_head=True)

    return model, tokenizer


# ─── SFT Training (supervised warmup) ─────────────────────────────

def train_sft(
    model,
    tokenizer,
    dataset: Dataset,
    formatter,
    output_dir: str,
    num_epochs: int = 2,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_seq_length: int = 512,
):
    """SFT warmup: teach the agent its role with supervised data."""
    from trl import SFTConfig

    # Format dataset
    def format_fn(examples):
        return [formatter(ex) for ex in examples]

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_seq_length=max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        formatting_func=format_fn,
    )
    trainer.train()
    return model


# ─── Simple Collaborative Training (SFT-based, no full RL) ────────

def train_collaborative(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "./outputs/collab",
    dataset_name: str = "trl-lib/tldr",
    dataset_size: int = 320,
    num_epochs: int = 3,
    lora_r: int = 8,
    lora_alpha: int = 16,
    use_4bit: bool = True,
    max_new_tokens: int = 256,
    learning_rate: float = 1e-4,
    task: str = "tldr",
):
    """Train two collaborative agents with QLoRA on a shared task.

    Agent A: concise summarizer / helper function writer
    Agent B: detailed summarizer / main function writer
    """
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if task == "coding":
        # Simple coding dataset
        questions = [
            "Check if a string is a palindrome",
            "Return the nth Fibonacci number (F(0)=0, F(1)=1)",
            "Count the number of vowels in a string",
            "Check if a number is prime",
            "Find the maximum element in a list",
            "Reverse a string",
            "Calculate factorial of n",
            "Check if two strings are anagrams",
        ]
        dataset = Dataset.from_dict({
            "question": questions,
            "prompt": questions,
        })
        formatters = build_code_formatters(tokenizer)
        reward_fn = execution_reward
    else:
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.select(range(min(dataset_size, len(dataset))))
        formatters = build_tldr_formatters(tokenizer)
        reward_fn = partial(
            length_ratio_reward, ratio_min=2.0, ratio_max=3.0, target_short_chars=220,
        )

    print(f"Dataset: {len(dataset)} samples")

    # Load two agents with LoRA
    print("Loading Agent A (concise/helper)...")
    agent_a, _ = load_agent_with_lora(
        model_name, lora_r=lora_r, lora_alpha=lora_alpha, use_4bit=use_4bit,
        device_map={"": 0},  # GPU 0
    )
    print("Loading Agent B (detailed/main)...")
    agent_b, _ = load_agent_with_lora(
        model_name, lora_r=lora_r, lora_alpha=lora_alpha, use_4bit=use_4bit,
        device_map={"": 1} if torch.cuda.device_count() > 1 else {"": 0},
    )

    agents = [agent_a, agent_b]
    optimizers = [
        torch.optim.AdamW(a.parameters(), lr=learning_rate) for a in agents
    ]
    device_a = next(agents[0].parameters()).device
    device_b = next(agents[1].parameters()).device

    os.makedirs(output_dir, exist_ok=True)

    # Format all prompts ahead of time
    formatted_data = []
    for ex in dataset:
        formatted_data.append({
            "prompt_a": formatters[0](ex),
            "prompt_b": formatters[1](ex),
        })

    B = 4  # batch size
    total_batches = len(formatted_data) // B + (1 if len(formatted_data) % B else 0)
    print(f"Starting MAAC training: {num_epochs} epochs, {len(formatted_data)} samples, batch={B}")

    from tqdm import tqdm

    for epoch in range(num_epochs):
        total_reward = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        pbar = tqdm(range(0, len(formatted_data), B), desc=f"Epoch {epoch+1}/{num_epochs}",
                    unit="batch", ncols=120)

        for start in pbar:
            batch = formatted_data[start:start + B]
            # Zero gradients before batch
            for opt in optimizers:
                opt.zero_grad()

            texts_a, texts_b = [], []
            tokens_a_list, tokens_b_list = [], []

            # --- Batch generate ---
            for d in batch:
                # Agent A
                inp_a = tokenizer(d["prompt_a"], return_tensors="pt").to(device_a)
                with torch.no_grad():
                    out_a = agents[0].generate(
                        **inp_a, max_new_tokens=max_new_tokens,
                        temperature=0.6, do_sample=True, top_p=0.9,
                    )
                text_a = tokenizer.decode(out_a[0][inp_a.input_ids.shape[1]:], skip_special_tokens=True)
                texts_a.append(text_a)
                tokens_a_list.append((inp_a, out_a))

                # Agent B
                full_b = f"{d['prompt_b']}\n\n[Reference]: {text_a}"
                inp_b = tokenizer(full_b, return_tensors="pt").to(device_b)
                with torch.no_grad():
                    out_b = agents[1].generate(
                        **inp_b, max_new_tokens=max_new_tokens,
                        temperature=0.6, do_sample=True, top_p=0.9,
                    )
                text_b = tokenizer.decode(out_b[0][inp_b.input_ids.shape[1]:], skip_special_tokens=True)
                texts_b.append(text_b)
                tokens_b_list.append((inp_b, out_b))

            # --- Batch rewards ---
            rewards = reward_fn(texts_a, texts_b)  # [B]
            for r in rewards:
                total_reward += r

            # --- MAAC update: actor + critic loss per sample, then step ---
            batch_actor = torch.tensor(0.0, requires_grad=True)
            batch_critic = 0.0

            for i in range(len(batch)):
                reward = rewards[i]

                # Agent A update
                actor_a, critic_a = _maac_update(
                    agents[0], tokenizer, tokens_a_list[i][0], tokens_a_list[i][1], reward,
                )
                # Agent B update
                actor_b, critic_b = _maac_update(
                    agents[1], tokenizer, tokens_b_list[i][0], tokens_b_list[i][1], reward,
                )
                batch_actor = batch_actor + actor_a + actor_b
                total_actor_loss += (actor_a.item() if isinstance(actor_a, torch.Tensor) else actor_a) + \
                                    (actor_b.item() if isinstance(actor_b, torch.Tensor) else actor_b)
                total_critic_loss += critic_a + critic_b

            # Step after accumulating batch gradients
            for opt in optimizers:
                opt.step()

            # Update tqdm postfix with current metrics
            pbar.set_postfix(
                reward=f"{sum(rewards)/len(rewards):.2f}",
                avg_r=f"{total_reward / (pbar.n * B):.3f}",
            )

        avg_reward = total_reward / len(formatted_data)
        avg_al = total_actor_loss / max(1, len(formatted_data) * 2)
        avg_cl = total_critic_loss / max(1, len(formatted_data) * 2)
        print(f"  ✓ avg_reward={avg_reward:.4f}, actor_loss={avg_al:.4f}, critic_loss={avg_cl:.4f}")

        # Save checkpoints
        agent_a.model.save_pretrained(os.path.join(output_dir, f"agent_a_epoch{epoch+1}"))
        agent_b.save_pretrained(os.path.join(output_dir, f"agent_b_epoch{epoch+1}"))

    # Save final
    final_a = os.path.join(output_dir, "agent_a_final")
    final_b = os.path.join(output_dir, "agent_b_final")
    agent_a.model.save_pretrained(final_a)
    agent_b.save_pretrained(final_b)
    print(f"Training complete. Models saved to {final_a} and {final_b}")
    return final_a, final_b


# ─── MAAC update (actor-critic) ────────────────────────────────────

def _maac_update(agent, tokenizer, inputs, gen_output, reward):
    """Single MAAC update: actor loss + critic loss.

    actor_loss  = -log_prob(completion|prompt) * advantage
    critic_loss = MSE(value, reward)
    advantage   = reward - value

    CausalLMWithValueHead.forward() returns both logits and values.
    """
    if reward == 0:
        return torch.tensor(0.0), 0.0

    device = next(agent.parameters()).device
    prompt_len = inputs.input_ids.shape[1]
    completion_ids = gen_output[0][prompt_len:]
    L = len(completion_ids)
    if L == 0:
        return torch.tensor(0.0), 0.0

    # Forward pass with gradients — CausalLMWithValueHead gives logits + values
    full_ids = torch.cat([inputs.input_ids[0], completion_ids]).unsqueeze(0).to(device)
    out = agent(full_ids, output_values=True)  # ActorCriticOutput

    # --- Actor loss: -log_prob * advantage ---
    logits = out.logits[0, prompt_len - 1 : -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_lps = log_probs[range(L), completion_ids.to(device)]
    seq_log_prob = token_lps.sum()

    # Advantage = reward - V(s)  where V(s) = value at last prompt position, averaged
    v_prompt = out.values[0, prompt_len - 1 : prompt_len + L]  # values over prompt tail + completion
    v_mean = v_prompt.mean() if v_prompt.numel() > 0 else torch.tensor(0.0, device=device)
    advantage = reward - v_mean.detach()

    actor_loss = -seq_log_prob * advantage

    # --- Critic loss: MSE(value, reward) ---
    reward_t = torch.tensor(reward, device=device, dtype=v_mean.dtype)
    critic_loss = torch.nn.functional.mse_loss(v_mean, reward_t)

    # --- Combined loss ---
    loss = actor_loss + 0.6 * critic_loss
    loss.backward()

    return actor_loss.detach(), critic_loss.item()


# ─── Baseline Inference Functions ──────────────────────────────────

def single_agent_inference(prompt: str, model, tokenizer, max_tokens=256) -> str:
    """Single agent does the whole task (B1)."""
    messages = [
        {"role": "system", "content": "Write a summary of this Reddit post."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def parallel_inference(prompt: str, model, tokenizer, max_tokens=256):
    """Two agents generate independently, no communication (B2)."""
    a = single_agent_inference(prompt + "\nWrite a one-sentence summary.", model, tokenizer, max_tokens // 2)
    b = single_agent_inference(prompt + "\nWrite a detailed summary.", model, tokenizer, max_tokens)
    return a, b


def sequential_inference(prompt: str, model, tokenizer, formatter_a, formatter_b, max_tokens=256):
    """A generates -> B generates based on A, base model (B3)."""
    prompt_a = formatter_a({"prompt": prompt})
    inputs_a = tokenizer(prompt_a, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_a = model.generate(**inputs_a, max_new_tokens=max_tokens // 2, temperature=0.1, do_sample=False)
    completion_a = tokenizer.decode(out_a[0][inputs_a.input_ids.shape[1]:], skip_special_tokens=True)
    prompt_b = formatter_b({"prompt": prompt}) + f"\n\n[Reference]: {completion_a}"
    inputs_b = tokenizer(prompt_b, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_b = model.generate(**inputs_b, max_new_tokens=max_tokens, temperature=0.1, do_sample=False)
    completion_b = tokenizer.decode(out_b[0][inputs_b.input_ids.shape[1]:], skip_special_tokens=True)
    return completion_a, completion_b


def discussion_inference(prompt: str, model, tokenizer, formatter_a, formatter_b, max_tokens=256):
    """One-round discussion: A -> B comments -> A revises (B4)."""
    prompt_a = formatter_a({"prompt": prompt})
    inputs_a = tokenizer(prompt_a, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_a = model.generate(**inputs_a, max_new_tokens=max_tokens // 2, temperature=0.1, do_sample=False)
    a1 = tokenizer.decode(out_a[0][inputs_a.input_ids.shape[1]:], skip_special_tokens=True)
    b_prompt = f"Review this summary and suggest improvements:\n{a1}"
    inputs_b = tokenizer(b_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_b = model.generate(**inputs_b, max_new_tokens=max_tokens // 2, temperature=0.1, do_sample=False)
    b_comment = tokenizer.decode(out_b[0][inputs_b.input_ids.shape[1]:], skip_special_tokens=True)
    revise_prompt = f"Revise your summary based on this feedback:\n{b_comment}"
    inputs_rev = tokenizer(revise_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_rev = model.generate(**inputs_rev, max_new_tokens=max_tokens, temperature=0.1, do_sample=False)
    a_revised = tokenizer.decode(out_rev[0][inputs_rev.input_ids.shape[1]:], skip_special_tokens=True)
    return a_revised, b_comment


# ─── Inference (single-turn collaboration) ────────────────────────

def collaborative_inference(
    prompt: str,
    agent_a,
    agent_b,
    tokenizer,
    formatter_a,
    formatter_b,
    max_new_tokens: int = 256,
) -> Dict[str, str]:
    """Run collaborative inference: Agent A → Agent B."""
    # Agent A
    prompt_a = formatter_a({"prompt": prompt})
    inputs_a = tokenizer(prompt_a, return_tensors="pt").to(agent_a.device)
    with torch.no_grad():
        out_a = agent_a.generate(
            **inputs_a, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False,
        )
    completion_a = tokenizer.decode(
        out_a[0][inputs_a.input_ids.shape[1]:], skip_special_tokens=True,
    )

    # Agent B (receives A's output)
    prompt_b = formatter_b({"prompt": prompt})
    full_prompt_b = f"{prompt_b}\n\n[Reference]: {completion_a}"
    inputs_b = tokenizer(full_prompt_b, return_tensors="pt").to(agent_b.device)
    with torch.no_grad():
        out_b = agent_b.generate(
            **inputs_b, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False,
        )
    completion_b = tokenizer.decode(
        out_b[0][inputs_b.input_ids.shape[1]:], skip_special_tokens=True,
    )

    return {"agent_a": completion_a, "agent_b": completion_b}


# ─── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train collaborative agents with QLoRA")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="./outputs/collab")
    parser.add_argument("--task", default="tldr", choices=["tldr", "coding"])
    parser.add_argument("--dataset_size", type=int, default=320)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--no_4bit", action="store_true")

    args = parser.parse_args()
    train_collaborative(
        model_name=args.model_name,
        output_dir=args.output_dir,
        task=args.task,
        dataset_size=args.dataset_size,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        use_4bit=not args.no_4bit,
    )
