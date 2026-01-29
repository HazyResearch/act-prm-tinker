"""
Text-only dataset constructor for policy gradient training
"""

from copy import copy
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


class DataCollatorForPolicyGradient(DataCollatorForLanguageModeling):
    """
    Custom collator for LM policy gradient training that pads log-probs and advantages.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> None:
        super().__init__(tokenizer=tokenizer, mlm=False, return_tensors=return_tensors, **kwargs)

    def __call__(self, features: list[dict[str, Any] | list[int] | Any]) -> dict[str, Any]:
        """
        Apply standard causal language modeling collator to HuggingFace Dataset features,
        but also pad log-probs and advantages / weights for policy gradient
        """
        # First let parent pad input_ids and attention_mask
        og_padding_side = copy(self.tokenizer.padding_side)
        self.tokenizer.padding_side = "right"
        parent_features = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features
        ]
        batch = super().__call__(parent_features)

        # Then handle logprobs, advantages / weights, and label_mask
        for key in ["logprobs", "advantages", "weights", "label_mask"]:
            if features[0].get(key, None) and key == "logprobs":
                dtype = torch.float
                padding_value = 1.0
            elif features[0].get(key, None):  # key in ["advantages", "weights"]
                dtype = torch.float
                padding_value = 0.0
            elif key == "label_mask":
                dtype = torch.bool
                padding_value = False
            else:
                continue
            
            values = [torch.tensor(f[key], dtype=dtype) for f in features]
            values = pad_sequence(values, batch_first=True, padding_value=padding_value)
            batch[key] = values

        # Reset padding side
        self.tokenizer.padding_side = og_padding_side
        return batch
