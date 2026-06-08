"""
Actor-Critic architecture for multi-agent LLM training.

Direct port from CoMLRL: comlrl/models/actor_critic.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass
class ActorCriticOutput:
    """Policy/value forward pass container."""

    logits: torch.Tensor
    values: Optional[torch.Tensor]
    hidden_states: Optional[Tuple[torch.Tensor, ...]]


class ValueHead(nn.Module):
    """Two-layer value head with optional hidden projection.

    hidden_states[-1] → Linear(hidden_dim) → Tanh → Linear(1)
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is not None and hidden_dim > 0:
            self.pre_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
            )
            last_dim = hidden_dim
        else:
            self.pre_projection = None
            last_dim = input_dim

        self.value_projection = nn.Linear(last_dim, 1)
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pre_projection is not None:
            hidden_states = self.pre_projection(hidden_states)
        return self.value_projection(hidden_states)


class CausalLMWithValueHead(nn.Module):
    """Wrap a causal LM with an optional learned value head.

    Forward returns ActorCriticOutput with logits AND values.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        value_head_hidden_dim: Optional[int] = None,
        attach_value_head: bool = True,
    ) -> None:
        super().__init__()
        self.model = base_model
        config = getattr(base_model, "config", None)
        if config is None:
            raise ValueError("Base model must provide a config with hidden size.")

        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Unsupported backbone: expected hidden_size or n_embd on config.")

        self.value_head: Optional[ValueHead] = None
        if attach_value_head:
            self.value_head = ValueHead(hidden_size, value_head_hidden_dim)
            base_params = list(base_model.parameters())
            if base_params:
                self.value_head.to(dtype=base_params[0].dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_values: bool = True,
        **kwargs,
    ) -> ActorCriticOutput:
        outputs: CausalLMOutputWithCrossAttentions = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        values: Optional[torch.Tensor] = None
        if output_values and self.value_head is not None:
            # Last hidden state → value for each position
            hidden = outputs.hidden_states[-1]
            values = self.value_head(hidden).squeeze(-1)

        return ActorCriticOutput(
            logits=outputs.logits,
            values=values,
            hidden_states=outputs.hidden_states,
        )

    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Proxy generation to underlying causal LM."""
        return self.model.generate(*args, **kwargs)
