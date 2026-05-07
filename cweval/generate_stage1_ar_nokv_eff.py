"""
CWEval stage-1 AR generation (local HF) with **KV cache disabled** for efficiency baselines.

Writes the same ``generated_*/**/*_raw.*`` layout as Dream stage-1 so
``evaluate_stage2.py`` can be reused.
"""

import sys
from pathlib import Path

_CWEVAL_ROOT = Path(__file__).resolve().parents[1]
if str(_CWEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CWEVAL_ROOT))

import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional

import fire
import torch
from p_tqdm import p_map
from tqdm import tqdm

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

_WORKER_HF: Dict[str, LocalHFAIAPI] = {}


def _gen_case(
    case: Dict[str, str],
    model_path: str,
    ppt: str,
    hf_kwargs: Dict[str, Any],
    eval_path: str,
    eff_tag: str,
) -> None:
    n = int(hf_kwargs.get("n", 1))
    missing: List[int] = []
    for i in range(n):
        if not os.path.exists(case["out_path_template"].format(index=i)):
            missing.append(i)
    if not missing:
        return

    prompt_text = GenerStage1Dream._render_prompt_text(
        ppt,
        case["lang"],
        case["code_prompt"],
        unsafe_code=case.get("unsafe_code", ""),
    )
    os.makedirs(os.path.dirname(case["prompt_path"]), exist_ok=True)
    with open(case["prompt_path"], "w", encoding="utf-8") as f:
        f.write(prompt_text)

    key = f"{model_path}|{json.dumps(hf_kwargs, sort_keys=True, default=str)}"
    api = _WORKER_HF.get(key)
    if api is None:
        init_kw = {k: v for k, v in hf_kwargs.items() if k not in ("n",)}
        api = LocalHFAIAPI(model_path, **init_kw)
        _WORKER_HF[key] = api
    pr = make_prompt(ppt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    resps = pr.req_ai(
        api,
        case["lang"],
        case["code_prompt"],
        metadata={k: v for k, v in case.items() if k not in ("code_prompt", "lang")},
        n=len(missing),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    if len(resps) != len(missing):
        raise ValueError("response count mismatch")

    for idx, text in zip(missing, resps):
        out = case["out_path_template"].format(index=idx)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    safe = case["task_file_path"].replace(os.sep, "_")[-180:]
    write_eff_metric_json(
        eval_path,
        f"ar_nokv_{eff_tag}_{safe}",
        {
            "eff_tag": eff_tag,
            "backend": "ar_local_hf",
            "use_cache": False,
            "task_file_path": case["task_file_path"],
            "wall_time_ms": round((t1 - t0) * 1000, 3),
            "n": len(missing),
            "max_completion_tokens": hf_kwargs.get("max_completion_tokens"),
        },
    )


class GenerARNokvEff:
    def __init__(
        self,
        eval_path: str = "",
        model_path: str = "",
        ppt: str = "direct",
        num_proc: int = 4,
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
        eff_tag: str = "ar_nokv",
    ):
        if not model_path:
            raise ValueError("model_path is required")
        self.model_path = model_path
        self.ppt = ppt
        self.num_proc = int(num_proc)
        self.langs = normalize_list_arg(langs)
        self.exclude_path = normalize_list_arg(exclude_path)
        self.include_path = normalize_list_arg(include_path)
        self.n = int(n)
        self.task_subset_file = (task_subset_file or "").strip()
        self.eff_tag = (eff_tag or "ar_nokv").strip() or "ar_nokv"
        if not eval_path:
            self.eval_path = os.path.join(
                "evals", f"eval_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_ar_nokv"
            )
        else:
            self.eval_path = eval_path
        os.makedirs(self.eval_path, exist_ok=True)
        # critical: no KV cache
        self.hf_kwargs = {
            "n": self.n,
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "use_cache": False,
            "dtype": dtype,
            "device_map": device_map,
        }
        self.cases = get_cweval_benchmark_cases(
            self.eval_path,
            self.langs,
            self.exclude_path,
            self.include_path,
        )
        if self.task_subset_file:
            self.cases = filter_cases_by_task_subset(self.cases, self.task_subset_file)
        self._write_config_snapshot()

    def _write_config_snapshot(self) -> None:
        path = os.path.join(self.eval_path, "generation_config_snapshot.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": "generation_only",
                    "backend": "ar_local_hf_no_kv",
                    "model_path": self.model_path,
                    "ppt": self.ppt,
                    "use_cache": False,
                    "ai_kwargs": self.hf_kwargs,
                    "num_cases": len(self.cases),
                    "eval_path": self.eval_path,
                    "task_subset_file": self.task_subset_file,
                    "eff_tag": self.eff_tag,
                },
                f,
                indent=2,
            )

    def gen(self) -> None:
        if not self.cases:
            print("No cases to generate.", flush=True)
            return
        cases_l = list(self.cases.values())
        p_map(
            _gen_case,
            cases_l,
            [self.model_path] * len(cases_l),
            [self.ppt] * len(cases_l),
            [self.hf_kwargs] * len(cases_l),
            [os.path.abspath(self.eval_path)] * len(cases_l),
            [self.eff_tag] * len(cases_l),
            num_cpus=self.num_proc,
        )
        # manifest (minimal, compatible with stage2 scan)
        manifest = {
            "eval_path": self.eval_path,
            "num_cases": len(self.cases),
            "exercise": "ar_nokv_eff",
        }
        with open(
            os.path.join(self.eval_path, "generation_manifest.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)
        print(f"Done. eval_path={self.eval_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(GenerARNokvEff)
