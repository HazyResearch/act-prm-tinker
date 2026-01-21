"""
Factory class for making TextWorld environments

Inspired by: https://github.com/balrog-ai/BALROG/blob/main/balrog/environments/textworld/base.py
"""

from __future__ import annotations

import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import textworld
from textworld.core import Environment as TextWorldEnvironment, Wrapper as TextWorldWrapper


@dataclass(frozen=True)
class TextWorldGameIndex:
    """
    Holds resolved game paths grouped by task.
    """
    games_by_task: dict[str, list[Path]]


class TextWorldFactory:
    """
    Singleton-like factory that:
    - scans *.ulx/*.z8
    - groups by parent folder name == task
    - creates a TextWorld env via `textworld.start(game_file, infos=EnvInfos(...))`
    """

    _instance: "TextWorldFactory | None" = None

    def __new__(cls, **kwargs: Any):
        """
        Create or choose existing instance of TextWorldFactory, e.g., so we can share
        one "factory" across multiple TextWorldEnv instances and handle file scanning once.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(**kwargs)
        return cls._instance

    def initialize(
        self,
        textworld_games_path: str | Path,
        tasks: list[str],
        # request infos
        objective: bool = True,
        description: bool = True,
        score: bool = True,
        max_score: bool = True,
        won: bool = True,
        # optionally include extras if you want
        **envinfos_kwargs: Any,
    ) -> None:
        """
        Initialize the TextWorldFactory
        """
        self.count = defaultdict(int)

        games_root = Path(textworld_games_path).expanduser().resolve()
        if not games_root.exists():
            raise FileNotFoundError(f"TextWorld games root not found: {games_root}")

        # Build EnvInfos request (TextWorld-native)
        self.request_infos = textworld.EnvInfos(
            objective=objective,
            description=description,
            score=score,
            max_score=max_score,
            won=won,
            **envinfos_kwargs,
        )

        # Scan game files
        games_by_task: dict[str, list[Path]] = defaultdict(list)
        for pattern in ("*.ulx", "*.z8"):
            for entry in sorted(glob.glob(str(games_root / "**" / pattern), recursive=True)):
                p = Path(entry)
                task = p.parent.name
                if task in tasks:
                    games_by_task[task].append(p)

        if not games_by_task:
            raise ValueError(
                f"No TextWorld games found under {games_root} for tasks={tasks} "
                "(expected .../<task_name>/*.ulx or *.z8)"
            )

        # Freeze index
        self.index = TextWorldGameIndex(games_by_task=dict(games_by_task))

    def list_tasks(self) -> list[str]:
        """
        List all available tasks
        """
        return sorted(self.index.games_by_task.keys())

    def num_games(self, task: str) -> int:
        """
        Get the number of games for a given task
        """
        if task not in self.index.games_by_task:
            raise KeyError(f"Unknown task '{task}'. Available: {self.list_tasks()}")
        return len(self.index.games_by_task[task])

    def select_game(self, task: str, sample_id: int | None = None) -> Path:
        """
        Select a game for a given task
        """
        if task not in self.index.games_by_task:
            raise KeyError(f"Unknown task '{task}'. Available: {self.list_tasks()}")

        games = self.index.games_by_task[task]
        if len(games) == 0:
            raise ValueError(f"No games registered for task '{task}'")

        if sample_id is not None:
            return games[sample_id % len(games)]  # wrap around if out of bounds

        # cycle if no seed provided
        self.count[task] += 1
        idx = self.count[task] % len(games)
        return games[idx]

    def make_env(
        self,
        task: str,
        sample_id: int | None = None,
    ) -> TextWorldEnvironment | TextWorldWrapper:
        """
        Make a TextWorld environment for a given task
        """
        game_file = self.select_game(task=task, sample_id=sample_id)
        # Create TextWorld environment
        # -> See https://github.com/microsoft/TextWorld/blob/809919eb96b78da05def8fca6d3c0ed1bc02efec/textworld/helpers.py#L19
        env = textworld.start(path=str(game_file), request_infos=self.request_infos)
        return env

    def __call__(self, task: str, sample_id: int | None = None) -> TextWorldEnvironment:
        """
        Alias for make_env
        """
        return self.make_env(task=task, sample_id=sample_id)
