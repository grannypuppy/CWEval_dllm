"""
Stage-1 generation only for Dream / Dream-multitask on CWEval.

This script intentionally only generates *_raw.* files and writes a manifest.
It does NOT run parse/compile/tests.
"""

import sys
from pathlib import Path

# Allow `python cweval/generate_stage1_dream.py ...` without PYTHONPATH=.
_CWEVAL_ROOT = Path(__file__).resolve().parents[1]
if str(_CWEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CWEVAL_ROOT))

import datetime
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import fire
from natsort import natsorted

from cweval.ai_dream import DreamAPI
from cweval.commons import BENCHMARK_DIR, LANGS
from cweval.ppt import make_prompt

_WORKER_API_CACHE: Dict[str, DreamAPI] = {}


class GenerStage1Dream:
    begin_prompt_anchor = "BEGIN PROMPT"
    begin_solution_anchor = "BEGIN SOLUTION"

    def __init__(
        self,
        eval_path: str = "",
        backend: str = "dream",
        model_path: str = "",
        codedllm_root: str = "",
        ppt: str = "direct",
        num_proc: int = 4,
        langs: List[str] = LANGS,
        exclude_path: List[str] = [],
        include_path: List[str] = [],
        # sampling params
        n: int = 20,
        max_completion_tokens: int = 2048,
        temperature: float = 0.8,
        # dream params
        steps: int = 256,
        top_p: float = 0.95,
        top_k: int = None,
        alg: str = "entropy",
        alg_temp: float = 0.1,
        threshold: float = None,
        use_rsp_prefix: bool = False,
        seed: int = None,
        gpu_ids: str = "",
        pin_worker_to_gpu: bool = True,
    ):
        if not model_path:
            raise ValueError("model_path is required for Dream stage-1 generation")

        self.backend = backend
        self.model_path = model_path
        self.codedllm_root = codedllm_root
        self.ppt = ppt
        self.num_proc = num_proc
        self.langs = self._normalize_list_arg(langs)
        self.exclude_path = self._normalize_list_arg(exclude_path)
        self.include_path = self._normalize_list_arg(include_path)
        self.num_samples = int(n)
        self.pin_worker_to_gpu = bool(pin_worker_to_gpu)
        self.gpu_ids = self._resolve_gpu_ids(gpu_ids)

        self.ai_kwargs = {
            "n": int(n),
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "steps": int(steps),
            "top_p": float(top_p),
            "top_k": top_k,
            "alg": alg,
            "alg_temp": float(alg_temp),
            "threshold": threshold,
            "use_rsp_prefix": bool(use_rsp_prefix),
            "seed": seed,
        }

        if not eval_path:
            self.eval_path = os.path.join(
                "evals", f"eval_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
            )
        else:
            self.eval_path = eval_path
            os.makedirs(self.eval_path, exist_ok=True)

        self.cases = self._get_cases()
        self._write_config_snapshot()

    def _write_config_snapshot(self) -> None:
        os.makedirs(self.eval_path, exist_ok=True)
        snapshot_path = os.path.join(self.eval_path, "generation_config_snapshot.json")
        payload = {
            "stage": "generation_only",
            "backend": self.backend,
            "model_path": self.model_path,
            "codedllm_root": self.codedllm_root,
            "ppt": self.ppt,
            "num_proc": self.num_proc,
            "langs": self.langs,
            "exclude_path": self.exclude_path,
            "include_path": self.include_path,
            "ai_kwargs": self.ai_kwargs,
            "gpu_ids": self.gpu_ids,
            "pin_worker_to_gpu": self.pin_worker_to_gpu,
            "num_cases": len(self.cases),
            "eval_path": self.eval_path,
        }
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _resolve_gpu_ids(gpu_ids: str) -> List[int]:
        if gpu_ids:
            return [int(x.strip()) for x in gpu_ids.split(",") if x.strip() != ""]
        try:
            import torch

            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except Exception:
            pass
        return []

    @staticmethod
    def _normalize_list_arg(value) -> List[str]:
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

    def _get_cases(self) -> Dict[str, Dict[str, str]]:
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
                if lang not in self.langs:
                    continue
                if any(exclude in task_file_path for exclude in self.exclude_path):
                    continue
                if self.include_path and not any(
                    include in task_file_path for include in self.include_path
                ):
                    continue

                with open(task_file_path, "r", encoding="utf-8") as f:
                    task_code = f.read()
                begin_solution_line_src = ""
                for line in task_code.splitlines():
                    if self.begin_solution_anchor in line:
                        begin_solution_line_src = line
                        break
                if not begin_solution_line_src:
                    raise ValueError(f"No solution anchor found in {task_file_path}")
                code_prompt = (
                    task_code.split(self.begin_prompt_anchor)[-1]
                    .split(begin_solution_line_src)[0]
                    .strip()
                )

                rel_task_file_path = os.path.relpath(task_file_path, BENCHMARK_DIR)
                out_path_template = os.path.join(
                    self.eval_path,
                    "generated_{index}",
                    rel_task_file_path.replace("_task", "_raw"),
                )
                cases[task_file_path] = {
                    "task_file_path": task_file_path,
                    "code_prompt": code_prompt,
                    "lang": lang,
                    "out_path_template": out_path_template,
                }
        return cases

    @staticmethod
    def _gen_case(
        backend: str,
        model_path: str,
        codedllm_root: str,
        ppt: str,
        case: Dict[str, str],
        ai_kwargs: Dict[str, Any],
        worker_gpu_ids: List[int],
        pin_worker_to_gpu: bool,
        rank: int,
    ) -> None:
        requested_n = int(ai_kwargs.get("n", 1))
        missing_indices = []
        for i in range(requested_n):
            out_path = case["out_path_template"].format(index=i)
            if not os.path.exists(out_path):
                missing_indices.append(i)
        if not missing_indices:
            print(f'{case["out_path_template"]} already completed, skipping', flush=True)
            return

        device = "cuda"
        if pin_worker_to_gpu and worker_gpu_ids:
            device = f"cuda:{worker_gpu_ids[rank % len(worker_gpu_ids)]}"

        cache_key = f"{backend}|{model_path}|{codedllm_root}|{device}|{json.dumps(ai_kwargs, sort_keys=True, default=str)}"
        aiapi = _WORKER_API_CACHE.get(cache_key)
        if aiapi is None:
            init_kwargs = dict(ai_kwargs)
            init_kwargs.pop("n", None)
            aiapi = DreamAPI(
                model_path=model_path,
                backend=backend,
                codedllm_root=codedllm_root or None,
                device=device,
                **init_kwargs,
            )
            _WORKER_API_CACHE[cache_key] = aiapi
        prompt = make_prompt(ppt)
        resps = prompt.req_ai(
            aiapi,
            case["lang"],
            case["code_prompt"],
            metadata={k: v for k, v in case.items() if k not in ["code_prompt", "lang"]},
            n=len(missing_indices),
        )
        if len(resps) != len(missing_indices):
            raise ValueError(
                f"Generated count mismatch: got {len(resps)}, expected {len(missing_indices)}"
            )

        for out_idx, resp in zip(missing_indices, resps):
            out_path = case["out_path_template"].format(index=out_idx)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(resp)

    def _build_manifest(self) -> Dict[str, Any]:
        entries = []
        complete_cases = 0
        for _, case in self.cases.items():
            missing = []
            existing = 0
            for i in range(self.num_samples):
                out_path = case["out_path_template"].format(index=i)
                if os.path.exists(out_path):
                    existing += 1
                else:
                    missing.append(i)
            is_complete = len(missing) == 0
            if is_complete:
                complete_cases += 1
            entries.append(
                {
                    "task_file_path": case["task_file_path"],
                    "lang": case["lang"],
                    "out_path_template": case["out_path_template"],
                    "expected_n": self.num_samples,
                    "existing_n": existing,
                    "missing_indices": missing,
                    "complete": is_complete,
                }
            )

        return {
            "stage": "generation_only",
            "eval_path": self.eval_path,
            "num_cases": len(self.cases),
            "num_complete_cases": complete_cases,
            "num_incomplete_cases": len(self.cases) - complete_cases,
            "entries": entries,
        }

    def write_manifest(self) -> None:
        manifest = self._build_manifest()
        manifest_path = os.path.join(self.eval_path, "generation_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(
            f"[manifest] complete={manifest['num_complete_cases']}/{manifest['num_cases']} "
            f"incomplete={manifest['num_incomplete_cases']} path={manifest_path}",
            flush=True,
        )

    def check(self) -> None:
        self.write_manifest()

    def gen(self) -> None:
        os.makedirs(self.eval_path, exist_ok=True)
        # Local CUDA + forked workers breaks PyTorch; use spawn for num_proc > 1.
        cases_list = list(self.cases.values())
        if self.num_proc <= 1:
            for rank, case in enumerate(cases_list):
                GenerStage1Dream._gen_case(
                    self.backend,
                    self.model_path,
                    self.codedllm_root,
                    self.ppt,
                    case,
                    self.ai_kwargs,
                    self.gpu_ids,
                    self.pin_worker_to_gpu,
                    rank,
                )
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=self.num_proc, mp_context=ctx
            ) as pool:
                futures = [
                    pool.submit(
                        GenerStage1Dream._gen_case,
                        self.backend,
                        self.model_path,
                        self.codedllm_root,
                        self.ppt,
                        case,
                        self.ai_kwargs,
                        self.gpu_ids,
                        self.pin_worker_to_gpu,
                        rank,
                    )
                    for rank, case in enumerate(cases_list)
                ]
                for fut in as_completed(futures):
                    fut.result()
        self.write_manifest()


if __name__ == "__main__":
    fire.Fire(GenerStage1Dream)

