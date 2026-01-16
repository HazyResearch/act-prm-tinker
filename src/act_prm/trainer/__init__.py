"""
Tinker trainers
"""
from typing import Any

from .act_prm import ActPrmTrainer
from .rl import RLTrainer
from .sft import SFTTrainer


def get_trainer(name: str, **kwargs: Any) -> ActPrmTrainer | RLTrainer | SFTTrainer:
    """
    Get a trainer by name
    """
    if name == "act_prm":
        return ActPrmTrainer(**kwargs)

    elif name == "rl":
        return RLTrainer(**kwargs)

    elif name == "sft":
        return SFTTrainer(**kwargs)
    
    else:
        raise NotImplementedError(f"Trainer {name} not implemented")


__all__ = [
    "get_trainer",
    "ActPrmTrainer",
    "RLTrainer",
    "SFTTrainer",
]