"""
CWEval stage-1: AR (local HF, no KV) with 1~R **feedback** rounds (``ast`` or ``bandit``).
Writes ``generated_*/**/*_raw`` compatible with ``evaluate_stage2.py``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fire
import torch

_CWEVAL_ROOT = Path(__file__).resolve().parents[1]
if str(_CWEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CWEVAL_ROOT))

import datetime

from cweval.efficiency_cweval_cases import (
    filter_cases_by_task_subset,
    get_cweval_benchmark_cases,
    normalize_list_arg,
    write_eff_metric_json,
)
from cweval.generate import LocalHFAIAPI
from cweval.commons import LANGS
from cweval.ppt import make_prompt
from cweval.generate_stage1_dream import GenerStage1Dream

import ast


def _extract_fenced_code(text: str, lang: str) -> str:
    if lang == "py" and "```python" in text:
        text = text.split("```python", 1)[-1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


def _ast_feedback(code: str) -> str:
    if not code.strip():
        return "empty code for AST check"
    try:
        ast.parse(code)
        return "AST parse OK (no syntax error); refine for functionality/security if needed."
    except SyntaxError as e:
        return f"AST/SyntaxError: {e}"


def _bandit_feedback(code: str) -> str:
    if not code.strip():
        return "empty code for bandit"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as t:
            t.write(code)
            path = t.name
        r = subprocess.run(
            [sys.executable, "-m", "bandit", "-q", "-f", "json", path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            os.unlink(path)
        except OSError:
            pass
    except OSError as e:  # noqa: BLE001
        return f"bandit: {e}"
    if r.returncode not in (0, 1):
        return f"bandit run failed: {r.stderr[:400]}"
    try:
        data = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return "bandit: could not parse JSON"
    res = (data.get("results") or [])
    if not res:
        return "bandit: no issues reported"
    it = res[0]
    sev, conf = it.get("issue_severity"), it.get("issue_confidence")
    tid, txt = it.get("test_id"), (it.get("issue_text") or "")[:500]
    return f"bandit issue [{sev}/{conf}] {tid}: {txt}"


class GenerARFeedbackEff:
    def __init__(
        self,
        eval_path: str = "",
        model_path: str = "",
        ppt: str = "direct",
        langs: List[str] = LANGS,
        exclude_path: List[str] = [],
        include_path: List[str] = [],
        n: int = 4,
        max_completion_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        task_subset_file: str = "",
        eff_tag: str = "ar_fb",
        num_rounds: int = 2,
        feedback_type: str = "ast",  # ast | bandit
        budget_mode: str = "per_round",  # per_round | strict
        max_chain_tokens: int = 512,  # strict: total new tokens per chain
    ):
        if not model_path:
            raise ValueError("model_path is required")
        if feedback_type not in ("ast", "bandit"):
            raise ValueError("feedback_type must be ast|bandit")
        if budget_mode not in ("per_round", "strict"):
            raise ValueError("budget_mode must be per_round|strict")
        if ppt != "direct" and int(num_rounds) > 1:
            print(
                "[Warning] num_rounds>1 only implemented for ppt=direct; will use 1 round.",
                flush=True,
            )
        self.ppt = ppt
        self.model_path = model_path
        self.langs = normalize_list_arg(langs)
        self.exclude_path = normalize_list_arg(exclude_path)
        self.include_path = normalize_list_arg(include_path)
        self.n = int(n)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_completion_tokens = int(max_completion_tokens)
        self.dtype = dtype
        self.device_map = device_map
        self.task_subset_file = (task_subset_file or "").strip()
        self.eff_tag = (eff_tag or "ar_fb").strip() or "ar_fb"
        self.num_rounds = max(1, int(num_rounds))
        self.feedback_type = feedback_type
        self.budget_mode = budget_mode
        self.max_chain_tokens = int(max_chain_tokens)
        if not eval_path:
            self.eval_path = os.path.join(
                "evals", f"eval_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_ar_fb"
            )
        else:
            self.eval_path = eval_path
        os.makedirs(self.eval_path, exist_ok=True)
        self.cases = get_cweval_benchmark_cases(
            self.eval_path, self.langs, self.exclude_path, self.include_path
        )
        if self.task_subset_file:
            self.cases = filter_cases_by_task_subset(self.cases, self.task_subset_file)
        self._api = LocalHFAIAPI(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device_map,
            n=1,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            use_cache=False,
        )
        self._write_config_snapshot()

    def _write_config_snapshot(self) -> None:
        path = os.path.join(self.eval_path, "generation_config_snapshot.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": "generation_only",
                    "backend": "ar_local_hf_feedback",
                    "model_path": self.model_path,
                    "ppt": self.ppt,
                    "use_cache": False,
                    "num_rounds": self.num_rounds,
                    "feedback_type": self.feedback_type,
                    "budget_mode": self.budget_mode,
                    "max_chain_tokens": self.max_chain_tokens,
                    "max_completion_tokens": self.max_completion_tokens,
                    "num_cases": len(self.cases),
                    "eff_tag": self.eff_tag,
                },
                f,
                indent=2,
            )

    def _tokens_for_round(self, r: int, R: int) -> int:
        if self.budget_mode == "per_round":
            return self.max_completion_tokens
        return max(1, self.max_chain_tokens // max(1, R))

    def _run_chain(
        self, case: Dict[str, str]
    ) -> Tuple[str, float, List[dict]]:
        R = self.num_rounds
        if self.ppt != "direct":
            pr = make_prompt(self.ppt)
            t0 = time.perf_counter()
            outs = pr.req_ai(
                self._api,
                case["lang"],
                case["code_prompt"],
                metadata={k: v for k, v in case.items() if k not in ("code_prompt", "lang")},
                n=1,
            )
            t1 = time.perf_counter()
            return (outs[0] if outs else ""), (t1 - t0), []

        base = GenerStage1Dream._render_prompt_text(
            self.ppt,
            case["lang"],
            case["code_prompt"],
            unsafe_code=case.get("unsafe_code", ""),
        )
        content = base
        total = 0.0
        trace: List[dict] = []
        last = ""
        for r in range(1, R + 1):
            mgen = self._tokens_for_round(r, R)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            parts = self._api.send_message(
                [{"role": "user", "content": content}],
                n=1,
                max_completion_tokens=mgen,
                temperature=self.temperature,
                top_p=self.top_p,
                use_cache=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            total += dt
            last = parts[0] if parts else ""
            trace.append(
                {
                    "round": r,
                    "wall_time_ms": round(dt * 1000, 3),
                    "max_new_tokens": mgen,
                }
            )
            if r < R:
                ext = _extract_fenced_code(last, case["lang"])
                if self.feedback_type == "ast":
                    fb = _ast_feedback(ext)
                else:
                    fb = _bandit_feedback(ext) if case["lang"] == "py" else _ast_feedback(
                        ext
                    )
                content = (
                    base
                    + f"\n\n## Previous model output (round {r})\n\n"
                    + (last[:8000] if last else "(empty)")
                    + "\n\n## Feedback to address before rewriting\n"
                    + fb
                    + "\n\nRegenerate: output a single complete fenced code block (same format as the original instruction).\n"
                )
        return last, total, trace

    def gen(self) -> None:
        n_out = self.n
        for case in self.cases.values():
            prompt_text = GenerStage1Dream._render_prompt_text(
                self.ppt,
                case["lang"],
                case["code_prompt"],
                unsafe_code=case.get("unsafe_code", ""),
            )
            os.makedirs(os.path.dirname(case["prompt_path"]), exist_ok=True)
            with open(case["prompt_path"], "w", encoding="utf-8") as f:
                f.write(prompt_text)
            for i in range(n_out):
                out_p = case["out_path_template"].format(index=i)
                if os.path.exists(out_p):
                    continue
                last, wall, trace = self._run_chain(case)
                os.makedirs(os.path.dirname(out_p), exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(last)
                safe = re.sub(r"[^\w.\-]+", "_", case["task_file_path"])[-150:]
                write_eff_metric_json(
                    self.eval_path,
                    f"ar_fb_{self.eff_tag}_{i}_{safe}",
                    {
                        "eff_tag": self.eff_tag,
                        "use_cache": False,
                        "feedback_type": self.feedback_type,
                        "num_rounds": self.num_rounds,
                        "task_file_path": case["task_file_path"],
                        "out_index": i,
                        "chain_wall_time_ms": round(wall * 1000, 3),
                        "round_trace": trace,
                    },
                )
        manifest = {
            "eval_path": self.eval_path,
            "num_cases": len(self.cases),
            "exercise": "ar_feedback_eff",
        }
        with open(
            os.path.join(self.eval_path, "generation_manifest.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(manifest, f, indent=2)
        print(f"Done. eval_path={self.eval_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(GenerARFeedbackEff)
