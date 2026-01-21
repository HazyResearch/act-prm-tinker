"""
Environments
"""

from typing import Any

from .base import Environment
from .types import EnvironmentState, EnvironmentStateWithAnswer


def get_env(name: str, is_async: bool = False, **kwargs: Any) -> Environment:
    """
    Get environment based on name
    """
    if name == "act_prm":
        if is_async:
            from .act_prm import AsyncActPrmEnv
            return AsyncActPrmEnv(**kwargs)
        else:
            from .act_prm import ActPrmEnv
            return ActPrmEnv(**kwargs)

    elif name == "act_prm_with_base_env":
        if is_async:
            from .act_prm import AsyncActPrmEnvWithBaseEnv
            return AsyncActPrmEnvWithBaseEnv(**kwargs)
        else:
            raise NotImplementedError(
                f"Sorry, synchronous version of '{name}' not implemented yet."
            )

    elif name == "textworld":
        if is_async:
            from .textworld import AsyncTextWorldEnv
            return AsyncTextWorldEnv(**kwargs)
        else:
            from .textworld import TextWorldEnv
            return TextWorldEnv(**kwargs)

    elif name == "hotpotqa_mc":
        if is_async:
            from .hotpotqa_mc import AsyncHotpotQAMultipleChoiceEnv
            return AsyncHotpotQAMultipleChoiceEnv(**kwargs)
        else:
            from .hotpotqa_mc import HotpotQAMultipleChoiceEnv
            return HotpotQAMultipleChoiceEnv(**kwargs)

    elif name == "browsecomp_plus_search":
        if is_async:
            from .browsecomp_plus import AsyncBrowseCompPlusSearchEnv
            return AsyncBrowseCompPlusSearchEnv(**kwargs)
        else:
            from .browsecomp_plus import BrowseCompPlusSearchEnv
            return BrowseCompPlusSearchEnv(**kwargs)

    elif name == "longbench_v2":
        if is_async:
            from .longbench_v2 import AsyncLongBenchEnvironment
            return AsyncLongBenchEnvironment(**kwargs)
        else:
            from .longbench_v2 import LongBenchEnvironment
            return LongBenchEnvironment(**kwargs)

    raise NotImplementedError(f"Sorry invalid environment: '{name}'.")


def load_env(name: str, **kwargs: Any) -> Environment:
    """
    Alias for get_env
    """
    return get_env(name, **kwargs)


__all__ = [
    "get_env",
    "load_env",
    "Environment",
    "EnvironmentState",
    "EnvironmentStateWithAnswer",
]
