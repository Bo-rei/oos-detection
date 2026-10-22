import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from src.models.components import ProjectionHead


class SmolLMExpert(nn.Module):
    def __init__(
        self,
        model_path: str,
        projection_dim: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 32,
        projection_bias: bool = False,
    ):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.base = get_peft_model(self.base, lora_cfg)
        hidden = getattr(self.base.config, "hidden_size", None)
        if hidden is None:
            raise RuntimeError("Base model missing hidden_size in config")
        self.projection = ProjectionHead(
            hidden, projection_dim=projection_dim, bias=projection_bias
        )

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
        proj = self.projection(pooled)
        features = F.normalize(proj, dim=1)
        return features
