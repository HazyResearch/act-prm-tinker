"""
Utils to find files
"""

from __future__ import annotations

import json
import os
from os.path import join
from pathlib import Path
from typing import Iterable, List, Union, Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rich import print as rich_print


TARGET_FILES = {
    "eval_gen_action_metrics.csv": "csv",
    "eval_lm_action_metrics.csv": "csv",
    "metrics.jsonl": "jsonl",
}

ARGN_TO_NAME = {
    "mibasi": "Mini-Batch Size",
    "lera": "LR",
    "se": "Seed",
    "re": "Replicate"
}


def get_argn(argname: str) -> str:
    """
    Get argument name as short string
    """
    return "".join([c[:2] for c in argname.split("_")])


def get_value_from_argv(argv: str, num_decimals: int = 0) -> str:
    """
    Get display value from argv portion in fname
    """
    is_float = False

    if argv[:2] == "0_":  # 0_001, 0_01, 0_1, etc.
        is_float = True
        val = float(argv.replace("_", "."))
    
    elif argv[1:3] == "e_":  # 1e_03, 4e_05, 1e_06, etc.
        is_float = True
        mult = float(argv[0])
        exp = float(argv[3:])
        val = mult * (0.1 ** exp)
    
    else:
        val = argv

    if is_float:
        # return f"{val:.{num_decimals}f}"
        return f"{val:.{num_decimals}e}"
    else:
        return val


def get_name_from_uid(unique_id: str, verbose: bool = False, join_sep: str = "-") -> str:
    if "acon=1" in unique_id:
        name = "Actions-only"
    else:
        name = "Thoughts + Actions"

    if verbose:  # Add other details
        details = []
        for argn_argv in unique_id.split(join_sep):
            argn, argv = argn_argv.split("=")
            if argn in ARGN_TO_NAME:
                details.append(f"{ARGN_TO_NAME[argn]}={get_value_from_argv(argv)}")
        name = f"{name} ({', '.join(details)})"

    return name



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
    """
    Define "non-empty" JSONL as: has at least 1 non-empty line.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    return True
        return False
    except FileNotFoundError:
        return False


def is_valid_metrics_file(path: Path, kind: str) -> bool:
    """
    Check if a file is a valid metrics file
    """
    if not path.exists() or not path.is_file():
        return False
    if kind == "csv":
        return is_nonempty_csv(path)
    if kind == "jsonl":
        return is_nonempty_jsonl(path)
    raise ValueError(f"Unknown kind: {kind}")


def unique_ids_from_strings(
    items: Iterable[Union[str, Path]],
    *,
    token_sep: str = "-",
    missing: str = "NA",
    join_sep: str = "-",
    sanitize: bool = True,
    always_keys: Iterable[str] = ("learning_rate", "mini_batch_size"),
) -> List[str]:
    """
    Create per-item identifiers using:
      - all kv keys whose values differ across items
      - PLUS any keys in always_keys (included regardless of uniqueness)
      - PLUS any non-kv tokens whose presence differs
    """
    # Convert keys to their filename versions
    always_keys = [get_argn(k) for k in always_keys]
    always_keys_set: Set[str] = set(always_keys)

    strings = [str(x) for x in items]
    if not strings:
        return []

    # --- 1) strip common prefix/suffix by path parts ---
    parts = [Path(s).parts for s in strings]
    min_len = min(len(p) for p in parts)

    pref = 0
    for i in range(min_len):
        if all(p[i] == parts[0][i] for p in parts):
            pref += 1
        else:
            break

    suf = 0
    for j in range(1, min_len - pref + 1):
        if all(p[-j] == parts[0][-j] for p in parts):
            suf += 1
        else:
            break

    cores = []
    for p in parts:
        core_parts = p[pref : (len(p) - suf if suf > 0 else len(p))]
        cores.append("/".join(core_parts))

    # --- 2) parse each core into kv dict + misc tokens ---
    kv_dicts: List[Dict[str, str]] = []
    misc_sets: List[Set[str]] = []

    for core in cores:
        toks = [t for t in core.split(token_sep) if t]
        kv: Dict[str, str] = {}
        misc: Set[str] = set()
        for t in toks:
            if "=" in t and t.count("=") == 1:
                k, v = t.split("=")
                kv[k] = v
            else:
                misc.add(t)
        kv_dicts.append(kv)
        misc_sets.append(misc)

    # --- 3) determine differing kv keys ---
    all_keys = sorted(set().union(*(d.keys() for d in kv_dicts)) | always_keys_set)

    differing_keys: Set[str] = set()
    for k in all_keys:
        vals = [(d.get(k, missing)) for d in kv_dicts]
        if len(set(vals)) > 1:
            differing_keys.add(k)

    # keys to include = differing OR always
    include_keys = [k for k in all_keys if (k in differing_keys) or (k in always_keys_set)]

    # --- 4) differing misc tokens (presence differs) ---
    all_misc = sorted(set().union(*misc_sets))
    differing_misc = []
    for t in all_misc:
        presence = [t in ms for ms in misc_sets]
        if len(set(presence)) > 1:
            differing_misc.append(t)

    # --- 5) build identifiers ---
    ids: List[str] = []
    for kv, ms in zip(kv_dicts, misc_sets):
        out_parts: List[str] = []
        for k in include_keys:
            out_parts.append(f"{k}={kv.get(k, missing)}")
        for t in differing_misc:
            if t in ms:
                out_parts.append(t)

        out = join_sep.join(out_parts) if out_parts else "all_same"
        if sanitize:
            out = (
                out.replace("/", "_")
                   .replace(" ", "")
                   .replace(":", "")
                   .replace("|", "_")
            )
        ids.append(out)

    # If still not unique, suffix duplicates deterministically
    seen: Dict[str, int] = {}
    final: List[str] = []
    for x in ids:
        c = seen.get(x, 0)
        seen[x] = c + 1
        final.append(x if c == 0 else f"{x}{join_sep}{c}")

    return final
