"""
PyTorch trainers for Hugging Face Transformer (PEFT) models
"""

from typing import Any

from .act_prm_for_sft import ActPrmForSftTrainer
from .rl import RLTrainer
from .sft import SftTrainer


def get_trainer(name: str, **kwargs: Any) -> ActPrmForSftTrainer | RLTrainer | SftTrainer:
    """
    Get a trainer by name
    """
    if name == "act_prm_for_sft":
        return ActPrmForSftTrainer(**kwargs)
    elif name == "rl":
        return RLTrainer(**kwargs)
    elif name == "sft":
        return SftTrainer(**kwargs)
    else:
        raise NotImplementedError(f"Trainer {name} not implemented")


__all__ = [
    "get_trainer",
    "ActPrmForSftTrainer",
    "RLTrainer",
    "SftTrainer",
]
