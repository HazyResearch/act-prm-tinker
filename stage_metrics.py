"""
Stage metrics in ./logs/ to Git

Dry run first (recommended):
```
uv run python stage_metrics.py --dry-run --verbose
```

Actually stage:
```
uv run python stage_metrics.py --verbose
```
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable

from rich import print as rich_print

TARGET_FILES = {
    "eval_gen_action_metrics.csv": "csv",
    "eval_lm_action_metrics.csv": "csv",
    "metrics.jsonl": "jsonl",
}


def is_nonempty_csv(path: Path) -> bool:
    """
    Define "non-empty" CSV as: has at least 2 non-empty lines
    (header + at least one data line).
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            nonempty_lines = 0
            for line in f:
                if line.strip():
                    nonempty_lines += 1
                    if nonempty_lines >= 2:
                        return True
        return False
    except FileNotFoundError:
        return False


def is_nonempty_jsonl(path: Path) -> bool:
    """Define "non-empty" JSONL as: has at least 1 non-empty line."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    return True
        return False
    except FileNotFoundError:
        return False

def should_stage(path: Path, kind: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if kind == "csv":
        return is_nonempty_csv(path)
    if kind == "jsonl":
        return is_nonempty_jsonl(path)
    raise ValueError(f"Unknown kind: {kind}")


def git_add(paths: Iterable[Path], repo_root: Path, dry_run: bool) -> None:
    paths = list(paths)
    if not paths:
        return

    # Use repo-root-relative paths for nicer output + safety.
    rel_paths = []
    for p in paths:
        try:
            rel_paths.append(str(p.relative_to(repo_root)))
        except ValueError:
            # If somehow outside repo root, fall back to absolute.
            rel_paths.append(str(p))

    if dry_run:
        for rp in rel_paths:
            _rp = f"[bright_blue]{rp}[/bright_blue]"
            rich_print(f"[bold yellow](dry-run)[/bold yellow] git add {_rp}")
        return

    # Batch add to reduce process overhead.
    cmd = ["git", "add", "--"] + rel_paths
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    for rp in rel_paths:
        _rp = f"[bright_blue]{rp}[/bright_blue]"
        rich_print(f"[bold green]staged:[/bold green] {_rp}")


def find_git_root(start: Path) -> Path:
    """
    Find git repo root by walking up until we see .git.
    """
    cur = start.resolve()
    for _ in range(50):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(f"Could not find .git repo root starting from: {start}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/scr/mzhang/projects/act-prm-tinker/logs/"),
        help="Base directory to search under.",
    )
    parser.add_argument(
        "--env_config",
        type=str,
        help="If specified, only stage metrics for the given environment config.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to git repo root (defaults to auto-detect from --base).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be staged without running git add.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print decisions for each candidate file.",
    )
    args = parser.parse_args()
    base = args.base.resolve()

    if args.env_config:
        base = Path(os.path.join(base, args.env_config))

    if not base.exists():
        raise SystemExit(f"Base path does not exist: {base}")

    repo_root = args.repo_root.resolve() if args.repo_root else find_git_root(base)

    to_stage: list[Path] = []

    # Walk directory tree. "stops when it finds files" interpreted as:
    # for each directory, if a target filename exists, evaluate it; no need
    # to do anything special beyond that.
    for root, _dirs, files in os.walk(base):
        files_set = set(files)
        for name, kind in TARGET_FILES.items():
            if name not in files_set:
                continue
            p = Path(root) / name
            ok = should_stage(p, kind)
            if args.verbose:
                _keep = "[bold]KEEP[/bold]"
                _skip = "[bold]skip[/bold]"
                # rich_print(f"{'KEEP' if ok else 'skip'}: {p} ({kind})")
                # print(f"{_keep if ok else _skip}: {p} ({kind})")
                if ok:
                    rich_print(f"[bright_cyan]{_keep}: {p} ({kind})[/bright_cyan]")
                else:
                    rich_print(f"[bright_red]{_skip}: {p} ({kind})[/bright_red]")
            if ok:
                to_stage.append(p)

    # De-dupe (just in case) while preserving order
    seen = set()
    unique_to_stage = []
    for p in to_stage:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique_to_stage.append(p)

    if not unique_to_stage:
        print("No non-empty metric files found to stage.")
        return

    git_add(unique_to_stage, repo_root=repo_root, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
