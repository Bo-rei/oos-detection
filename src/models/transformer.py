"""Shared CausalLM loading and masked pooling for trainable heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


class LoRACausalEncoder(nn.Module):
    """Owns a LoRA-wrapped CausalLM while preserving the historic ``base`` key."""

    def _load_base(
        self,
        model_path: str,
        *,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> int:
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.base = get_peft_model(
            self.base,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            ),
        )
        hidden_size = getattr(self.base.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Base model missing hidden_size in config")
        return int(hidden_size)

    def _pooled_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None
            else outputs.hidden_states[-1]
        )
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        return torch.sum(hidden * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )
