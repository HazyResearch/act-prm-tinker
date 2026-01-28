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
    
    else:
        raise NotImplementedError(f"Generator {name} not implemented")


__all__ = [
    "get_generator_constructor",
    "HuggingFaceGenerator",
]
