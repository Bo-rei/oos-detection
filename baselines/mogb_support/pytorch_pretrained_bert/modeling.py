"""Minimal model_pipeline modeling surface for the frozen MOGB official clone.

This is an API-compatibility shim over ``transformers``. It intentionally
preserves only the model_pipeline pieces that the official MOGB code imports.
"""

from __future__ import annotations

import inspect

from transformers import BertConfig as BertConfig
from transformers import BertModel as _TransformersBertModel
from transformers import BertPreTrainedModel as _TransformersBertPreTrainedModel
from transformers.utils import CONFIG_NAME as CONFIG_NAME
from transformers.utils import WEIGHTS_NAME as WEIGHTS_NAME


class BertPreTrainedModel(_TransformersBertPreTrainedModel):
    """Restore the model_pipeline ``init_bert_weights`` helper used by MOGB."""

    all_tied_weights_keys = {}

    def __init__(self, config) -> None:
        super().__init__(config)
        self.all_tied_weights_keys = {}

    def init_bert_weights(self, module) -> None:
        self._init_weights(module)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        init_params = inspect.signature(cls.__init__).parameters
        if "num_labels" in kwargs and "num_labels" in init_params:
            model_args = (*model_args, kwargs.pop("num_labels"))
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class BertModel(_TransformersBertModel):
    """Restore the model_pipeline forward contract from pytorch-pretrained-bert."""

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_all_encoded_layers,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        if output_all_encoded_layers:
            # Legacy callers expect one tensor per encoder block, not the
            # embedding-state entry included by modern hidden_states.
            encoder_layers = list(outputs.hidden_states[1:])
            return encoder_layers, pooled_output
        return outputs.last_hidden_state, pooled_output


__all__ = [
    "BertConfig",
    "BertModel",
    "BertPreTrainedModel",
    "CONFIG_NAME",
    "WEIGHTS_NAME",
]
