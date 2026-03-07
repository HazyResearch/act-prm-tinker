"""
Generate training reports from W&B run data.

Pulls metrics from W&B, creates clean matplotlib plots, and generates
a markdown report with embedded plots, run metadata, and summary stats.

Usage:
    uv run python scripts/wandb_report.py --project act-prm-cc
    uv run python scripts/wandb_report.py --project act-prm-cc --run-id abc123
    uv run python scripts/wandb_report.py --project act-prm-cc --active-only
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

logger = logging.getLogger(__name__)

# Academic-style plot config
PLOT_STYLE = {
    "figure.figsize": (7, 4),
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "legend.fontsize": 10,
    "legend.framealpha": 0.8,
}

# Group metrics by prefix for multi-line plots
METRIC_GROUPS = {
    "loss": ["train/loss", "eval/loss"],
    "reward": ["eval/final_reward", "train/final_reward"],
    "accuracy": ["eval/accuracy", "train/accuracy"],
}

# Metrics to skip (internal/progress)
SKIP_PATTERNS = ["_runtime", "_timestamp", "_step", "_wandb", "progress/"]

COLORS = ["#2171b5", "#cb181d", "#238b45", "#6a3d9a", "#ff7f00", "#a65628"]


def short_run_name(run: wandb.apis.public.Run) -> str:
    """Derive a short, human-readable name from the run config."""
    config = run.config
    env = config.get("env_config", "")
    # Extract task + difficulty from env_config path
    # e.g., "act_lm/tw_coin_easy" -> "coin-easy"
    match = re.search(r"tw_(\w+?)_(easy|medium|hard)", env)
    if match:
        task, diff = match.groups()
        short = f"{task}-{diff}"
    else:
        short = env.replace("/", "-").replace("_", "-")

    rep = config.get("replicate", 0)
    seed = config.get("seed", 0)
    return f"{short}-r{rep}-s{seed}"


def should_skip(key: str) -> bool:
    return any(p in key for p in SKIP_PATTERNS)


def plot_metric(
    history: list[dict],
    keys: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    """Plot one or more metrics on the same axes. Returns True if data was plotted."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots()
        plotted = False

        for i, key in enumerate(keys):
            steps, vals = [], []
            for row in history:
                if key in row and row[key] is not None:
                    step = row.get("_step", len(steps))
                    steps.append(step)
                    vals.append(row[key])
            if not steps:
                continue
            label = key.split("/")[-1] if "/" in key else key
            color = COLORS[i % len(COLORS)]
            ax.plot(steps, vals, color=color, label=label, alpha=0.9)
            plotted = True

        if not plotted:
            plt.close(fig)
            return False

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(keys) > 1:
            ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return True


def get_latest_values(history: list[dict], keys: list[str]) -> dict[str, float | None]:
    """Get the most recent non-None value for each key."""
    latest = {}
    for key in keys:
        val = None
        for row in reversed(history):
            if key in row and row[key] is not None:
                val = row[key]
                break
        latest[key] = val
    return latest


def generate_report(
    run: wandb.apis.public.Run,
    reports_dir: Path,
    entity: str,
) -> Path:
    """Generate a full report for a single W&B run."""
    name = short_run_name(run)
    run_dir = reports_dir / name
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Fetch full history
    history = list(run.scan_history())
    if not history:
        logger.warning("No history for run %s", run.id)
        return run_dir

    # Discover all metric keys
    all_keys = set()
    for row in history:
        all_keys.update(k for k in row if not should_skip(k))

    # Plot grouped metrics
    plotted_keys = set()
    plot_files = []

    for group_name, group_keys in METRIC_GROUPS.items():
        present = [k for k in group_keys if k in all_keys]
        if not present:
            continue
        out = plots_dir / f"{group_name}.png"
        ylabel = group_name.replace("_", " ").title()
        if plot_metric(history, present, ylabel, ylabel, out):
            plot_files.append((group_name, out.relative_to(run_dir)))
            plotted_keys.update(present)

    # Plot remaining ungrouped metrics individually
    for key in sorted(all_keys - plotted_keys):
        safe_name = key.replace("/", "_").replace(" ", "_")
        out = plots_dir / f"{safe_name}.png"
        ylabel = key.split("/")[-1].replace("_", " ").title()
        title = key.replace("/", " / ").replace("_", " ").title()
        if plot_metric(history, [key], title, ylabel, out):
            plot_files.append((key, out.relative_to(run_dir)))

    # Get summary stats
    latest = get_latest_values(history, list(all_keys))
    config = run.config
    total_steps = max((r.get("_step", 0) for r in history), default=0)

    # Build markdown
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    started = run.created_at if hasattr(run, "created_at") else "?"
    wandb_url = run.url

    md = []
    md.append(f"# Training Report: {name}")
    md.append("")
    md.append(f"> Auto-generated {now} from [W&B run]({wandb_url})")
    md.append("")

    # Metadata table
    md.append("## Run Metadata")
    md.append("")
    md.append("| Field | Value |")
    md.append("|-------|-------|")
    md.append(f"| **Run ID** | `{run.id}` |")
    md.append(f"| **Status** | {run.state} |")
    md.append(f"| **Started** | {started} |")
    md.append(f"| **Steps** | {total_steps} |")

    # Key config values
    for cfg_key in ["env_config", "eval_env_config", "model_config", "lora_config",
                     "trainer_config", "learning_rate", "mini_batch_size",
                     "gradient_accumulation_steps", "seed", "replicate", "data_seed",
                     "group_size", "hide_observations", "actions_only"]:
        if cfg_key in config:
            md.append(f"| **{cfg_key}** | `{config[cfg_key]}` |")
    md.append("")

    # Latest metrics summary
    md.append("## Latest Metrics")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    for key in sorted(all_keys):
        val = latest.get(key)
        if val is not None:
            if isinstance(val, float):
                md.append(f"| {key} | {val:.6f} |")
            else:
                md.append(f"| {key} | {val} |")
    md.append("")

    # Plots
    md.append("## Training Curves")
    md.append("")
    for label, rel_path in plot_files:
        display = label.replace("/", " / ").replace("_", " ").title()
        md.append(f"### {display}")
        md.append("")
        md.append(f"![{display}]({rel_path})")
        md.append("")

    # Write report
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(md))
    logger.info("Report written to %s (%d plots)", report_path, len(plot_files))
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Generate training reports from W&B")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default="hazy-research", help="W&B entity")
    parser.add_argument("--run-id", default=None, help="Specific run ID (default: all)")
    parser.add_argument("--active-only", action="store_true", help="Only running/crashed runs")
    parser.add_argument("--reports-dir", default="./reports", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reports_dir = Path(args.reports_dir) / args.project
    api = wandb.Api()

    if args.run_id:
        run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
        out = generate_report(run, reports_dir, args.entity)
        print(f"Report: {out / 'report.md'}")
    else:
        filters = {}
        if args.active_only:
            filters = {"state": {"$in": ["running", "crashed"]}}
        runs = api.runs(f"{args.entity}/{args.project}", filters=filters)
        for run in runs:
            try:
                out = generate_report(run, reports_dir, args.entity)
                print(f"Report: {out / 'report.md'}")
            except Exception as e:
                logger.error("Failed for run %s: %s", run.id, e)


if __name__ == "__main__":
    main()
