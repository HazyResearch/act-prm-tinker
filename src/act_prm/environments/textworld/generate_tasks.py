"""
Generate TextWorld games

Example commands:
```
uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir /scr/mzhang/data/textworld/tw_games/ \
--task treasure_hunter \
--n 10 \
--seed_start 0 \
--level 20

uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir /scr/mzhang/data/textworld/tw_games/ \
--task cooking_game \
--n 10 \
--seed_start 0 \
--recipe 2 \
--go 6 \
--open --cook --cut
```

Note: for cooking game, we do need to do a package-level patch. 

Currently, in `.venv/lib/python3.12/site-packages/textworld/challenges/tw_cooking/cooking.py`
L1016 has:

```python
nb_distractors = abs(int(rng_objects.randn(1) * 3 + nb_ingredients))
```

With newer numpy versions, this will raise a scalar issue. We should instead change it to:

```python
nb_distractors = abs(int(rng_objects.randn() * 3 + nb_ingredients))
```
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


TASK_CHOICES = [
    "treasure_hunter", "coin_collector", "cooking_game",
    "treasure_hunter_unseen", "treasure_hunter_newtools", "treasure_hunter_shifted"
]


def run(cmd: list[str]) -> None:
    """
    Print and run a bash command
    """
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def gen_treasure(out_dir: Path, level: int, seed: int) -> None:
    """
    Generate a treasure hunter game
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tw_treasure_lvl{level:03d}_seed{seed:03d}.z8"
    run([
        "tw-make", "tw-treasure_hunter",
        "--level", str(level),
        "--seed", str(seed),
        "--output", str(out),
        "--format", "z8",
        "-f", "--silent",
    ])


def gen_coin(out_dir: Path, level: int, seed: int) -> None:
    """
    Generate a coin collector game
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tw_coin_lvl{level:03d}_seed{seed:03d}.z8"
    run([
        "tw-make", "tw-coin_collector",
        "--level", str(level),
        "--seed", str(seed),
        "--output", str(out),
        "--format", "z8",
        "-f", "--silent",
    ])


def gen_cooking(
    out_dir: Path,
    seed: int,
    recipe: int,
    go: int,
    open_: bool,
    cook: bool,
    cut: bool,
    drop: bool,
    split: str,
) -> None:
    """
    Generate a cooking game
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tw_cook_r{recipe:03d}_go{go:03d}_seed{seed:03d}_{split}.z8"
    cmd = [
        "tw-make", "tw-cooking",
        "--recipe", str(recipe),
        "--go", str(go),
        "--seed", str(seed),
        "--split", split,
        "--output", str(out),
        "--format", "z8",
        "-f", "--silent",
    ]
    if open_:
        cmd.append("--open")
    if cook:
        cmd.append("--cook")
    if cut:
        cmd.append("--cut")
    if drop:
        cmd.append("--drop")
    run(cmd)


def main() -> None:
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, required=True)

    parser.add_argument("--task", choices=TASK_CHOICES, required=True)

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)

    # treasure/coin
    parser.add_argument("--level", type=int, default=10)

    # cooking knobs
    parser.add_argument("--recipe", type=int, default=2)
    parser.add_argument("--go", type=int, default=9)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--cook", action="store_true")
    parser.add_argument("--cut", action="store_true")
    parser.add_argument("--drop", action="store_true")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")

    args = parser.parse_args()
    out_dir = args.cache_dir / args.task

    num_generated = 0
    i = 0
    while num_generated < args.n:
        seed = args.seed_start + i
        try:
            if args.task.startswith("treasure_hunter"):
                gen_treasure(out_dir, level=args.level, seed=seed)
            elif args.task.startswith("coin_collector"):
                gen_coin(out_dir, level=args.level, seed=seed)
            else:
                gen_cooking(
                    out_dir=out_dir,
                    seed=seed,
                    recipe=args.recipe,
                    go=args.go,
                    open_=args.open,
                    cook=args.cook,
                    cut=args.cut,
                    drop=args.drop,
                    split=args.split,
                )
            num_generated += 1
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating task {i}: {e}")
        i += 1


if __name__ == "__main__":
    main()
