"""
Build CWEval benchmark task dicts for stage-1 generation (Dream or AR).

Structure matches ``cweval.generate_stage1_dream.GenerStage1Dream._get_cases``:
each value has task_file_path, code_prompt, unsafe_code, lang, out_path_template, prompt_path.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from natsort import natsorted

from cweval.commons import BENCHMARK_DIR

BEGIN_PROMPT = "BEGIN PROMPT"
BEGIN_SOLUTION = "BEGIN SOLUTION"


def get_cweval_benchmark_cases(
    eval_path: str,
    langs: List[str],
    exclude_path: List[str],
    include_path: List[str],
) -> Dict[str, Dict[str, str]]:
    cases: Dict[str, Dict[str, str]] = {}
    for root, _, files in os.walk(BENCHMARK_DIR):
        if "__pycache__" in root:
            continue
        for file in natsorted(files):
            file_wo_ext, ext = os.path.splitext(file)
            task_file_path = os.path.join(root, file)
            lang = ext[1:]
            if not (ext and file_wo_ext.endswith("_task")):
                continue
            if lang not in langs:
                continue
            if any(ex in task_file_path for ex in exclude_path):
                continue
            if include_path and not any(inc in task_file_path for inc in include_path):
                continue
            with open(task_file_path, "r", encoding="utf-8") as f:
                task_code = f.read()
            begin_solution_line_src = ""
            for line in task_code.splitlines():
                if BEGIN_SOLUTION in line:
                    begin_solution_line_src = line
                    break
            if not begin_solution_line_src:
                raise ValueError(f"No solution anchor in {task_file_path}")
            code_prompt = (
                task_code.split(BEGIN_PROMPT)[-1]
                .split(begin_solution_line_src)[0]
                .strip()
            )
            unsafe_code = ""
            unsafe_file_path = task_file_path.replace("_task.", "_unsafe.")
            if os.path.exists(unsafe_file_path):
                with open(unsafe_file_path, "r", encoding="utf-8") as _f:
                    unsafe_raw = _f.read()
                _entrypoint_anchor = "BEGIN ENTRYPOINT"
                _lines = unsafe_raw.splitlines()
                _cutoff = len(_lines)
                for _i, _line in enumerate(_lines):
                    if _entrypoint_anchor in _line:
                        _cutoff = _i
                        break
                unsafe_code = "\n".join(_lines[:_cutoff]).strip()

            rel_task_file_path = os.path.relpath(task_file_path, BENCHMARK_DIR)
            out_path_template = os.path.join(
                eval_path,
                "generated_{index}",
                rel_task_file_path.replace("_task", "_raw"),
            )
            prompt_path = os.path.join(
                eval_path,
                "prompts",
                rel_task_file_path.replace("_task", "_prompt.txt"),
            )
            cases[task_file_path] = {
                "task_file_path": task_file_path,
                "code_prompt": code_prompt,
                "unsafe_code": unsafe_code,
                "lang": lang,
                "out_path_template": out_path_template,
                "prompt_path": prompt_path,
            }
    return cases


def filter_cases_by_task_subset(
    cases: Dict[str, Dict[str, str]], task_subset_file: str
) -> Dict[str, Dict[str, str]]:
    from pathlib import Path

    path = Path(task_subset_file)
    if not path.is_file():
        raise FileNotFoundError(task_subset_file)
    needles: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                needles.append(s)
    if not needles:
        return cases
    out: Dict[str, Dict[str, str]] = {}
    for k, v in cases.items():
        if any(n in k for n in needles):
            out[k] = v
    return out


def normalize_list_arg(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return [str(value)]


def write_eff_metric_json(
    eval_path: str,
    name_suffix: str,
    payload: Dict[str, Any],
) -> None:
    import json
    import os

    mdir = os.path.join(eval_path, "eff_metrics")
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"{name_suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
