import torch.nn as nn
import torch

from src.models.transformer import LoRACausalEncoder


class QwenRouter(LoRACausalEncoder):
    def __init__(
        self,
        model_path: str,
        num_classes: int = 10,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ):
        super().__init__()
        hidden_size = self._load_base(
            model_path, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1
        )
        self.score = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        return self.score(self._pooled_features(input_ids, attention_mask))


class SmolLMRouter(LoRACausalEncoder):
    def __init__(
        self,
        model_path: str,
        num_classes: int = 10,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ):
        super().__init__()
        hidden_size = self._load_base(
            model_path, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(self._pooled_features(input_ids, attention_mask))
