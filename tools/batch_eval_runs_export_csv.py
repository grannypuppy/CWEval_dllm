#!/usr/bin/env python3
"""
Batch CWEval stage-2 evaluation and export pass@k metrics for every run under ``evals/``.

Typical usage (mirror ``run.sh``: Podman/Docker, then full pipeline + reports)::

    cd /path/to/CWEval
    export PYTHONPATH=.
    export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
    # optional: CPATH / LIBRARY_PATH for host-side compile if not using Docker for tests

    python tools/batch_eval_runs_export_csv.py \\
        --full-pipeline \\
        --num-proc 8 \\
        --docker true \\
        --ks 1,2,3,4 \\
        --output evals/pass_at_summary.csv

Light mode (default if ``--full-pipeline`` is off): for each run that has
``res_all.json``, run ``stage2_report --k …`` for every ``--ks`` value so
``report_pass_at_{k}.txt`` is (re)generated from merged results, then write CSV.
Runs **without** ``res_all.json`` are skipped with a warning — use
``--full-pipeline`` for those.

Explicit ``--export-only`` is the same light path (no parse/compile/test); it
only refuses to be combined with ``--full-pipeline``.  Use ``--ks`` to choose
which ``report_pass_at_{k}.txt`` files are regenerated and merged into the CSV.

Add ``--continue-on-error`` if one run fails but you want the rest + CSV anyway.

Fire CLI reminder for a single run::

    python cweval/evaluate_stage2.py --eval_path evals/foo --num_proc 8 \\
        stage2_pipeline --docker True --k 4
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

# -----------------------------------------------------------------------------
# Report parsing (aligned with current ``report_pass_at_{k}.txt`` format)
# -----------------------------------------------------------------------------

_PASS_HEADER = re.compile(r"^pass@(\d+)\s+(.+?)\s*$")
# ``func-sec`` must be tried before ``func`` (left-first alternation is safe here
# because ``func`` does not prefix-match ``func-sec`` at token boundary).
_METRIC_LINE = re.compile(r"^(func-sec|func|sec)@(\d+)\s+(.+)$")


def parse_report_pass_at_file(path: Path) -> List[Dict[str, Any]]:
    """
    Parse one ``report_pass_at_{k}.txt`` into a list of blocks:
    ``{k, slice, func, sec, func_sec}``.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "=" * 16:
            if current and "slice" in current:
                blocks.append(current)
            current = None
            continue
        m_pass = _PASS_HEADER.match(line)
        if m_pass:
            if current and "slice" in current:
                blocks.append(current)
            current = {
                "k": int(m_pass.group(1)),
                "slice": m_pass.group(2).strip(),
                "func": "",
                "sec": "",
                "func_sec": "",
            }
            continue
        m_met = _METRIC_LINE.match(line)
        if m_met and current is not None:
            key, _k_str, val = m_met.group(1), m_met.group(2), m_met.group(3).strip()
            if key == "func":
                current["func"] = val
            elif key == "sec":
                current["sec"] = val
            elif key == "func-sec":
                current["func_sec"] = val

    if current and "slice" in current:
        blocks.append(current)
    return blocks


def discover_eval_dirs(evals_root: Path) -> List[Path]:
    """Direct children of ``evals/`` that look like a CWEval run."""
    if not evals_root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(evals_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        has_gen = (p / "generated_0").is_dir()
        has_res = (p / "res_all.json").is_file()
        if has_gen or has_res:
            out.append(p)
    return out


def build_evaluate_cmd(
    cweval_root: Path,
    eval_rel: str,
    num_proc: int,
    subcommand: str,
    *,
    docker: Optional[bool] = None,
    k: Optional[int] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        str(cweval_root / "cweval" / "evaluate_stage2.py"),
        "--eval_path",
        eval_rel,
        "--num_proc",
        str(num_proc),
        subcommand,
    ]
    if subcommand == "stage2_pipeline":
        cmd.extend(["--docker", "true" if docker else "false"])
        if k is not None:
            cmd.extend(["--k", str(k)])
    elif subcommand == "stage2_report":
        if k is not None:
            cmd.extend(["--k", str(k)])
    return cmd


def _slice_to_column_tag(slice_raw: str) -> str:
    """
    Turn report slice labels into a safe CSV column fragment.

    Examples:
        ``all`` -> ``all``
        ``core/c/`` -> ``core_c``
        ``core/cpp/`` -> ``core_cpp``
        ``core/py/`` -> ``core_py``
        ``lang/c`` -> ``lang_c``
    """
    s = slice_raw.strip().strip("/")
    s = s.replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _build_wide_rows(
    long_rows: List[Dict[str, str]],
    ks: List[int],
    all_run_names: List[str],
) -> tuple[List[Dict[str, str]], List[str]]:
    """
    One row per run in ``all_run_names``; columns ``k{K}_all_func``, ``k{K}_all_sec``,
    ``k{K}_all_func_sec`` from long rows where ``slice == 'all'``.
    """
    bucket: DefaultDict[str, Dict[int, Dict[str, str]]] = defaultdict(dict)
    for r in long_rows:
        if r.get("slice", "").strip() != "all":
            continue
        run = r["run_name"]
        try:
            k = int(r["k"])
        except ValueError:
            continue
        bucket[run][k] = {
            "func": r.get("func", ""),
            "sec": r.get("sec", ""),
            "func_sec": r.get("func_sec", ""),
        }

    wide_fields = ["run_name"]
    for k in ks:
        wide_fields.extend(
            [f"k{k}_all_func", f"k{k}_all_sec", f"k{k}_all_func_sec"]
        )

    wide_rows: List[Dict[str, str]] = []
    run_names = sorted(set(all_run_names), key=str.lower)
    for run_name in run_names:
        row: Dict[str, str] = {"run_name": run_name}
        for k in ks:
            m = bucket[run_name].get(k, {})
            row[f"k{k}_all_func"] = m.get("func", "")
            row[f"k{k}_all_sec"] = m.get("sec", "")
            row[f"k{k}_all_func_sec"] = m.get("func_sec", "")
        wide_rows.append(row)
    return wide_rows, wide_fields


def _build_extra_wide_rows(
    long_rows: List[Dict[str, str]],
    ks: List[int],
    all_run_names: List[str],
) -> tuple[List[Dict[str, str]], List[str]]:
    """
    One row per run; columns for **every** slice tag::

        k{K}_{tag}_func, k{K}_{tag}_sec, k{K}_{tag}_func_sec

    where ``tag`` is from :func:`_slice_to_column_tag` (``all``, ``core_c``,
    ``core_py``, ``lang_c``, …).  Column order: ``all`` first, then remaining
    tags sorted lexicographically.
    """
    # run -> k -> tag -> metrics
    bucket: DefaultDict[str, DefaultDict[int, Dict[str, Dict[str, str]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    seen_tags: set[str] = set()

    for r in long_rows:
        raw_slice = (r.get("slice") or "").strip()
        if not raw_slice:
            continue
        tag = _slice_to_column_tag(raw_slice)
        seen_tags.add(tag)
        run = r["run_name"]
        try:
            k = int(r["k"])
        except ValueError:
            continue
        bucket[run][k][tag] = {
            "func": r.get("func", ""),
            "sec": r.get("sec", ""),
            "func_sec": r.get("func_sec", ""),
        }

    tag_order = sorted(seen_tags, key=lambda t: (0 if t == "all" else 1, t))

    wide_fields = ["run_name"]
    for k in ks:
        for tag in tag_order:
            wide_fields.extend(
                [
                    f"k{k}_{tag}_func",
                    f"k{k}_{tag}_sec",
                    f"k{k}_{tag}_func_sec",
                ]
            )

    wide_rows: List[Dict[str, str]] = []
    run_names = sorted(set(all_run_names), key=str.lower)
    for run_name in run_names:
        row: Dict[str, str] = {"run_name": run_name}
        for k in ks:
            by_tag = bucket[run_name].get(k, {})
            for tag in tag_order:
                m = by_tag.get(tag, {})
                row[f"k{k}_{tag}_func"] = m.get("func", "")
                row[f"k{k}_{tag}_sec"] = m.get("sec", "")
                row[f"k{k}_{tag}_func_sec"] = m.get("func_sec", "")
        wide_rows.append(row)
    return wide_rows, wide_fields


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool) -> int:
    printable = " ".join(cmd)
    print(f"[run] {printable}", flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + prev if prev else "")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return proc.returncode


def main() -> int:
    script_path = Path(__file__).resolve()
    default_cweval = script_path.parent.parent

    ap = argparse.ArgumentParser(
        description="Batch stage2 + export pass@k rows for all eval runs under evals/."
    )
    ap.add_argument(
        "--cweval-root",
        type=Path,
        default=default_cweval,
        help="CWEval repo root (default: parent of tools/).",
    )
    ap.add_argument(
        "--eval-subdir",
        type=str,
        default="evals",
        help="Subdirectory under cweval-root containing runs (default: evals).",
    )
    ap.add_argument(
        "--include-regex",
        type=str,
        default="",
        help="If set, only run names matching this regex (re.search).",
    )
    ap.add_argument(
        "--exclude-regex",
        type=str,
        default="",
        help="If set, skip run names matching this regex.",
    )
    ap.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run stage2_pipeline (parse/compile/test/merge + default report sweep) once per run.",
    )
    ap.add_argument(
        "--pipeline-k",
        type=int,
        default=None,
        help="If set with --full-pipeline, pass this k to the final report step only (Fire: stage2_pipeline --k K).",
    )
    ap.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="num_proc for evaluate_stage2.py (default: 8).",
    )
    ap.add_argument(
        "--docker",
        type=str,
        default="true",
        choices=("true", "false"),
        help="stage2_pipeline --docker value (default: true).",
    )
    ap.add_argument(
        "--ks",
        type=str,
        default="1,2,3,4",
        help="Comma-separated k values for stage2_report + CSV columns (default: 1,2,3,4).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Long-format CSV path (one row per run × k × language slice).",
    )
    ap.add_argument(
        "--output-wide",
        type=Path,
        default=None,
        help="Optional wide CSV: one row per run; columns k{K}_all_* only (slice ``all``).",
    )
    ap.add_argument(
        "--output-extra-wide",
        type=Path,
        default=None,
        help=(
            "Optional extra-wide CSV: one row per run; columns k{K}_{tag}_func/sec/func_sec "
            "for every slice (e.g. k4_core_py_func, k4_lang_c_sec, k4_all_func_sec)."
        ),
    )
    ap.add_argument(
        "--export-only",
        action="store_true",
        help=(
            "Do not run stage2_pipeline. For each run with res_all.json, run "
            "stage2_report for each k in --ks (writes report_pass_at_{k}.txt), "
            "then export CSV. Same as default when --full-pipeline is not set."
        ),
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="On subprocess failure, log and continue with remaining runs (exit 1 if any failed).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = ap.parse_args()

    if args.full_pipeline and args.export_only:
        print(
            "ERROR: use either --full-pipeline or --export-only, not both.",
            file=sys.stderr,
        )
        return 2

    cweval_root: Path = args.cweval_root.resolve()
    evals_root = (cweval_root / args.eval_subdir).resolve()
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if not ks:
        print("ERROR: --ks must list at least one integer.", file=sys.stderr)
        return 2

    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    runs = discover_eval_dirs(evals_root)
    filtered: List[Path] = []
    for p in runs:
        name = p.name
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        filtered.append(p)

    if not filtered:
        print(f"No eval runs found under {evals_root}", file=sys.stderr)
        return 1

    docker_bool = args.docker == "true"
    eval_rel_prefix = args.eval_subdir.rstrip("/")
    any_failure = False

    for run_dir in filtered:
        rel = f"{eval_rel_prefix}/{run_dir.name}"
        if args.full_pipeline:
            cmd = build_evaluate_cmd(
                cweval_root,
                rel,
                args.num_proc,
                "stage2_pipeline",
                docker=docker_bool,
                k=args.pipeline_k,
            )
            rc = run_cmd(cmd, cweval_root, args.dry_run)
            if rc != 0:
                print(
                    f"ERROR: pipeline failed for {rel} (exit {rc})",
                    file=sys.stderr,
                )
                any_failure = True
                if not args.dry_run and not args.continue_on_error:
                    return rc

        res_all = run_dir / "res_all.json"
        if not res_all.is_file():
            print(
                f"WARN: skip {rel}: no res_all.json — cannot run stage2_report; "
                f"use --full-pipeline to produce it.",
                file=sys.stderr,
            )
            continue

        for k in ks:
            cmd = build_evaluate_cmd(
                cweval_root,
                rel,
                args.num_proc,
                "stage2_report",
                k=k,
            )
            rc = run_cmd(cmd, cweval_root, args.dry_run)
            if rc != 0:
                print(
                    f"ERROR: stage2_report k={k} failed for {rel} (exit {rc})",
                    file=sys.stderr,
                )
                any_failure = True
                if not args.dry_run and not args.continue_on_error:
                    return rc

    # Collect CSV rows
    rows: List[Dict[str, str]] = []
    for run_dir in filtered:
        run_name = run_dir.name
        for k in ks:
            report_path = run_dir / f"report_pass_at_{k}.txt"
            blocks = parse_report_pass_at_file(report_path)
            if not blocks:
                rows.append(
                    {
                        "run_name": run_name,
                        "k": str(k),
                        "slice": "",
                        "func": "",
                        "sec": "",
                        "func_sec": "",
                        "report_file": str(report_path.relative_to(cweval_root))
                        if report_path.is_file()
                        else "",
                    }
                )
                continue
            for b in blocks:
                rows.append(
                    {
                        "run_name": run_name,
                        "k": str(b["k"]),
                        "slice": b["slice"],
                        "func": b["func"],
                        "sec": b["sec"],
                        "func_sec": b["func_sec"],
                        "report_file": f"{eval_rel_prefix}/{run_name}/report_pass_at_{k}.txt",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_name", "k", "slice", "func", "sec", "func_sec", "report_file"]
    if not args.dry_run:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[done] Wrote {len(rows)} rows to {args.output.resolve()}", flush=True)
    else:
        print(f"[dry-run] Would write {len(rows)} rows to {args.output}", flush=True)

    if args.output_wide and not args.dry_run:
        wide_rows, wide_fields = _build_wide_rows(
            rows, ks, [p.name for p in filtered]
        )
        args.output_wide.parent.mkdir(parents=True, exist_ok=True)
        with args.output_wide.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(wide_rows)
        print(
            f"[done] Wrote {len(wide_rows)} wide rows to {args.output_wide.resolve()}",
            flush=True,
        )
    elif args.output_wide and args.dry_run:
        print(f"[dry-run] Would also write wide CSV to {args.output_wide}", flush=True)

    if args.output_extra_wide and not args.dry_run:
        xw_rows, xw_fields = _build_extra_wide_rows(
            rows, ks, [p.name for p in filtered]
        )
        args.output_extra_wide.parent.mkdir(parents=True, exist_ok=True)
        with args.output_extra_wide.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=xw_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(xw_rows)
        print(
            f"[done] Wrote {len(xw_rows)} extra-wide rows to "
            f"{args.output_extra_wide.resolve()}",
            flush=True,
        )
    elif args.output_extra_wide and args.dry_run:
        print(
            f"[dry-run] Would also write extra-wide CSV to {args.output_extra_wide}",
            flush=True,
        )

    return 1 if any_failure and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
