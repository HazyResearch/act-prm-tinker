"""
Helper functions for Action Language Modeling environments
"""

import torch
from transformers import PreTrainedTokenizerBase
from rich import print as rich_print
from copy import copy

ROYGBIV = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]


def check_model_inputs(
    batch: dict[str, torch.Tensor],
    hf_tokenizer: PreTrainedTokenizerBase | None = None,
    run_url: str | None = None,
    run_cmd: str | None = None,
) -> None:
    """
    Sanity-check model inputs by rich printing them
    """
    decoded_inputs = hf_tokenizer.batch_decode(batch["input_ids"][:, 1:])
    _labels = copy(batch["labels"][:, 1:])
    _labels[_labels == -100] = 0  # -100 will cause tokenization errors
    decoded_labels = hf_tokenizer.batch_decode(_labels)
    for idx, decoded_input in enumerate(decoded_inputs):
        rich_print(f"[cyan]Input {idx}:\n{decoded_input}\n[/cyan]")
        rich_print(f"[green]Label {idx}:\n{decoded_labels[idx]}\n[/green]")
        rich_print("=" * 100)
    # Keep run url and cmd in display
    if run_url is not None:
        rich_print(f"[bold]Run url:[link={run_url}]{run_url}[/link][/bold]")
    if run_cmd is not None:
        rich_print(f"[bold]Run cmd:[bright_cyan]{run_cmd}[/bright_cyan][/bold]")
