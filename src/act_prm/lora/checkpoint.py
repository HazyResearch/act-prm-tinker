"""
LoRA checkpointing helper functions
"""

from collections import OrderedDict
from os.path import join
from typing import Any

import torch
from safetensors.torch import load_file

from peft import get_peft_model_state_dict, set_peft_model_state_dict, PeftModel


# -----------------
# Default / One GPU
# -----------------
def save_trainable_weights(model: Any) -> OrderedDict:
    """
    Save all trainable weights (i.e., LoRA weights) of a standard nn.Module model

    If saved to `state_dict`, should load later with:
    `model.load_state_dict(state_dict, strict=False)`
    """
    with torch.no_grad():
        state_dict = OrderedDict()
        for n, p in model.named_parameters():
            if p.requires_grad:
                state_dict[n] = p.cpu()
    return state_dict


def save_lora(model: Any, out_dir: str):
    """
    Save a model with LoRA weights
    """
    model.save_pretrained(out_dir)


def load_lora(model: Any, checkpoint_path: str, is_trainable: bool = True) -> Any:
    """
    Load LoRA weights from checkpoint to a model
    """
    if isinstance(model, PeftModel):
        _checkpoint_path = join(checkpoint_path, "adapter_model.safetensors")
        _adapter_name = model.active_adapter
        state = load_file(_checkpoint_path)  # state_dict-like mapping from safetensors
        set_peft_model_state_dict(model, state, adapter_name=_adapter_name)
    else:
        model = PeftModel.from_pretrained(
            model, checkpoint_path, is_trainable=is_trainable
        )
    if is_trainable:
        model.train()
    return model
