"""
Hugging Face Transformers generators for PyTorch-based training
"""

from functools import partial
from typing import Any, Callable

from .base import HuggingFaceGenerator


def get_generator_constructor(name: str, **kwargs: Any) -> Callable[..., HuggingFaceGenerator]:
    """
    Get a (partially initialized) Hugging Face Generator constructor by name
    """
    if name == "default":
        return partial(HuggingFaceGenerator, **kwargs)

    elif name == "action_prompt_act_prm":
        from .act_prm import ActionPromptActPrmGenerator
        return partial(ActionPromptActPrmGenerator, **kwargs)
    
    else:
        raise NotImplementedError(f"Generator {name} not implemented")


__all__ = [
    "get_generator_constructor",
    "HuggingFaceGenerator",
]
