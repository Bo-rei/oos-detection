import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


class QwenRouter(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_classes: int = 10,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.base = get_peft_model(self.base, lora_cfg)
        hidden_size = getattr(self.base.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Base model missing hidden_size in config")
        self.score = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_embeddings = torch.sum(hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        logits = self.score(pooled)
        return logits


class SmolLMRouter(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_classes: int = 10,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.base = get_peft_model(self.base, lora_cfg)
        hidden_size = getattr(self.base.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Base model missing hidden_size in config")
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )
        return self.classifier(pooled)
