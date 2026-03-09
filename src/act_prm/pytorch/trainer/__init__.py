"""
PyTorch trainers for Hugging Face Transformer (PEFT) models
"""

from typing import Any

from .act_prm_for_sft import ActPrmForSftTrainer
from .act_prm_joint import ActPrmJointTrainer
from .rl import RLTrainer
from .sft import SftTrainer
from .sft_rl import SftRlTrainer


def get_trainer(
    name: str, **kwargs: Any
) -> ActPrmForSftTrainer | ActPrmJointTrainer | RLTrainer | SftTrainer | SftRlTrainer:
    """
    Get a trainer by name
    """
    if name == "act_prm_for_sft":
        return ActPrmForSftTrainer(**kwargs)
    elif name == "act_prm_joint":
        return ActPrmJointTrainer(**kwargs)
    elif name == "rl":
        return RLTrainer(**kwargs)
    elif name == "sft":
        return SftTrainer(**kwargs)
    elif name == "sft_rl":
        return SftRlTrainer(**kwargs)
    else:
        raise NotImplementedError(f"Trainer {name} not implemented")


__all__ = [
    "get_trainer",
    "ActPrmForSftTrainer",
    "ActPrmJointTrainer",
    "RLTrainer",
    "SftRlTrainer",
    "SftTrainer",
]
