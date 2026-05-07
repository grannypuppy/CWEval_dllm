"""
Stage-1 generation only for Dream / Dream-multitask on CWEval (efficiency runs).

Same backends and manifest behavior as ``generate_stage1_dream.py``, plus:

- ``--task_subset_file``: only benchmark tasks whose path contains a line from the file
- ``--eff_tag``: tag written into ``eff_metrics/*.json`` records
- Per-task wall time logged under ``<eval_path>/eff_metrics/`` (one JSON per task)
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
import queue
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import fire
from natsort import natsorted
from tqdm import tqdm

from cweval.ai_dream import DreamAPI
from cweval.commons import BENCHMARK_DIR, LANGS
from cweval.ppt import make_prompt

_WORKER_API_CACHE: Dict[str, DreamAPI] = {}


class GenerStage1DreamEff:
    begin_prompt_anchor = "BEGIN PROMPT"
    begin_solution_anchor = "BEGIN SOLUTION"

    #: Backends supported by DreamAPI (validated at DreamAPI init time).
    VALID_BACKENDS = (
        "dream",
        "dream_auto",
        "dreamcoder",
        "llada",
        "dream_multitask",
        "dream_bandit",
        "dream_bandit_conf",
        "dream_multitask_bandit_conf",
        "dream_ast",
        "dream_ast_strt",
        "dream_multitask_ast",
        "dream_codeql",
    )

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
        bandit_penalty_ratio: float = 0.5,
        bandit_conf_penalty_ratio: float = 0.5,
        bandit_every_n_steps: int = 1,
        bandit_timeout_sec: float = 4.0,
        use_rsp_prefix: bool = False,
        seed: int = None,
        gpu_ids: str = "",
        test_limit: int = 0,
        task_subset_file: str = "",
        eff_tag: str = "",
    ):
        if not model_path:
            raise ValueError("model_path is required for Dream stage-1 generation")
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                f"Choose from {self.VALID_BACKENDS}."
            )

        self.backend = backend
        self.model_path = model_path
        self.codedllm_root = codedllm_root
        self.ppt = ppt
        self.num_proc = num_proc
        self.langs = self._normalize_list_arg(langs)
        self.exclude_path = self._normalize_list_arg(exclude_path)
        self.include_path = self._normalize_list_arg(include_path)
        self.num_samples = int(n)
        self.gpu_ids = self._resolve_gpu_ids(gpu_ids)
        self.test_limit = int(test_limit)
        self.task_subset_file = (task_subset_file or "").strip()
        self.eff_tag = (eff_tag or "").strip()

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
            "bandit_penalty_ratio": float(bandit_penalty_ratio),
            "bandit_conf_penalty_ratio": float(bandit_conf_penalty_ratio),
            "bandit_every_n_steps": int(bandit_every_n_steps),
            "bandit_timeout_sec": float(bandit_timeout_sec),
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
        if self.test_limit > 0:
            self.cases = dict(list(self.cases.items())[: self.test_limit])
        if self.task_subset_file:
            self.cases = self._filter_cases_by_subset(self.cases, self.task_subset_file)
        self.ai_kwargs["__eff_eval_path__"] = os.path.abspath(self.eval_path)
        self.ai_kwargs["__eff_tag__"] = self.eff_tag
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
            "test_limit": self.test_limit,
            "num_cases": len(self.cases),
            "eval_path": self.eval_path,
            "task_subset_file": self.task_subset_file,
            "eff_tag": self.eff_tag,
        }
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _render_prompt_text(
        ppt: str, lang: str, code_prompt: str, unsafe_code: str = ""
    ) -> str:
        prompt_cls = make_prompt(ppt)
        if hasattr(prompt_cls, "PPT") and hasattr(prompt_cls, "LANG_INSTR"):
            fmt_kwargs: dict = dict(
                lang=lang,
                lang_instr=prompt_cls.LANG_INSTR[lang],
                code_prompt=code_prompt,
            )
            # VulPATCH-style templates require the vulnerable code snippet.
            if "{unsafe_code}" in prompt_cls.PPT:
                fmt_kwargs["unsafe_code"] = unsafe_code
            return prompt_cls.PPT.format(**fmt_kwargs)
        # Fallback for unknown prompt implementations.
        return code_prompt

    @staticmethod
    def _resolve_gpu_ids(gpu_ids) -> List[int]:
        # Fire may pass ``0`` as int; ``if gpu_ids:`` would treat 0 as falsy and
        # wrongly fall back to all GPUs. Treat explicit int (including 0) as one id.
        if gpu_ids is None:
            gpu_ids = ""
        if isinstance(gpu_ids, int):
            return [gpu_ids]
        if isinstance(gpu_ids, str) and gpu_ids.strip():
            return [int(x.strip()) for x in gpu_ids.split(",") if x.strip() != ""]
        if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) > 0:
            out: List[int] = []
            for item in gpu_ids:
                if isinstance(item, str):
                    out.extend(
                        [int(x.strip()) for x in item.split(",") if x.strip() != ""]
                    )
                else:
                    out.append(int(item))
            return out
        try:
            import torch

            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except Exception:
            pass
        return []

    @staticmethod
    def _filter_cases_by_subset(
        cases: Dict[str, Dict[str, str]], subset_path: str
    ) -> Dict[str, Dict[str, str]]:
        path = Path(subset_path)
        if not path.is_file():
            raise FileNotFoundError(f"task_subset_file not found: {subset_path}")
        needles = []
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

    @staticmethod
    def _append_eff_metric(
        eval_path: str,
        eff_tag: str,
        backend: str,
        task_file_path: str,
        wall_time_s: float,
        ai_kwargs: Dict[str, Any],
        num_responses: int,
    ) -> None:
        if not eval_path:
            return
        metrics_dir = os.path.join(eval_path, "eff_metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        safe = (
            task_file_path.replace(os.sep, "_")
            .replace("/", "_")
            .replace("..", "_")
        )[-200:]
        fname = f"{safe}.json"
        rec = {
            "eff_tag": eff_tag,
            "backend": backend,
            "task_file_path": task_file_path,
            "wall_time_ms": round(wall_time_s * 1000.0, 3),
            "steps": ai_kwargs.get("steps"),
            "n_requested": ai_kwargs.get("n"),
            "num_responses": num_responses,
            "max_completion_tokens": ai_kwargs.get("max_completion_tokens"),
        }
        with open(os.path.join(metrics_dir, fname), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

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

                # Load the paired _unsafe.* file so VulPATCH-style prompts can
                # include the vulnerable implementation as in-context reference.
                unsafe_code = ""
                unsafe_file_path = task_file_path.replace("_task.", "_unsafe.")
                if os.path.exists(unsafe_file_path):
                    with open(unsafe_file_path, "r", encoding="utf-8") as _f:
                        unsafe_raw = _f.read()
                    # Strip the entrypoint (main / test harness) and below so
                    # the prompt only shows the vulnerable function body.
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
                    self.eval_path,
                    "generated_{index}",
                    rel_task_file_path.replace("_task", "_raw"),
                )
                prompt_path = os.path.join(
                    self.eval_path,
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

    @staticmethod
    def _gen_case(
        backend: str,
        model_path: str,
        codedllm_root: str,
        ppt: str,
        case: Dict[str, str],
        ai_kwargs: Dict[str, Any],
        device: str,
        progress_queue: Optional[Any] = None,
    ) -> None:
        requested_n = int(ai_kwargs.get("n", 1))
        missing_indices = []
        for i in range(requested_n):
            out_path = case["out_path_template"].format(index=i)
            if not os.path.exists(out_path):
                missing_indices.append(i)
        if not missing_indices:
            print(f'{case["out_path_template"]} already completed, skipping', flush=True)
            if progress_queue is not None:
                progress_queue.put(1)
            return

        prompt_text = GenerStage1DreamEff._render_prompt_text(
            ppt, case["lang"], case["code_prompt"],
            unsafe_code=case.get("unsafe_code", ""),
        )
        os.makedirs(os.path.dirname(case["prompt_path"]), exist_ok=True)
        with open(case["prompt_path"], "w", encoding="utf-8") as f:
            f.write(prompt_text)

        eff_eval_path = str(ai_kwargs.get("__eff_eval_path__") or "")
        eff_tag = str(ai_kwargs.get("__eff_tag__") or "")

        init_kwargs = {k: v for k, v in ai_kwargs.items() if not k.startswith("__eff_")}
        cache_key = f"{backend}|{model_path}|{codedllm_root}|{device}|{json.dumps(init_kwargs, sort_keys=True, default=str)}"
        aiapi = _WORKER_API_CACHE.get(cache_key)
        if aiapi is None:
            kw = dict(init_kwargs)
            kw.pop("n", None)
            aiapi = DreamAPI(
                model_path=model_path,
                backend=backend,
                codedllm_root=codedllm_root or None,
                device=device,
                **kw,
            )
            _WORKER_API_CACHE[cache_key] = aiapi
        prompt = make_prompt(ppt)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        t0 = time.perf_counter()
        resps = prompt.req_ai(
            aiapi,
            case["lang"],
            case["code_prompt"],
            metadata={k: v for k, v in case.items() if k not in ["code_prompt", "lang"]},
            n=len(missing_indices),
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        t1 = time.perf_counter()
        GenerStage1DreamEff._append_eff_metric(
            eff_eval_path,
            eff_tag,
            backend,
            case["task_file_path"],
            t1 - t0,
            init_kwargs,
            len(resps),
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
        if progress_queue is not None:
            progress_queue.put(1)

    @staticmethod
    def _run_shard(
        backend: str,
        model_path: str,
        codedllm_root: str,
        ppt: str,
        shard_cases: List[Dict[str, str]],
        ai_kwargs: Dict[str, Any],
        device: str,
        progress_queue: Optional[Any] = None,
    ) -> None:
        for case in shard_cases:
            GenerStage1DreamEff._gen_case(
                backend=backend,
                model_path=model_path,
                codedllm_root=codedllm_root,
                ppt=ppt,
                case=case,
                ai_kwargs=ai_kwargs,
                device=device,
                progress_queue=progress_queue,
            )

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
        cases_list = list(self.cases.values())
        if not cases_list:
            self.write_manifest()
            return

        worker_count = max(1, min(self.num_proc, len(cases_list)))
        shards: List[List[Dict[str, str]]] = [[] for _ in range(worker_count)]
        for idx, case in enumerate(cases_list):
            shards[idx % worker_count].append(case)

        if self.gpu_ids:
            devices = [f"cuda:{self.gpu_ids[i % len(self.gpu_ids)]}" for i in range(worker_count)]
        else:
            devices = ["cuda"] * worker_count
        total_tasks = len(cases_list)

        if worker_count == 1:
            for case in tqdm(shards[0], total=len(shards[0]), desc="stage1 tasks"):
                GenerStage1DreamEff._gen_case(
                    backend=self.backend,
                    model_path=self.model_path,
                    codedllm_root=self.codedllm_root,
                    ppt=self.ppt,
                    case=case,
                    ai_kwargs=self.ai_kwargs,
                    device=devices[0],
                )
        else:
            ctx = mp.get_context("spawn")
            manager = ctx.Manager()
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
                future_sizes = {}
                for i in range(worker_count):
                    fut = pool.submit(
                        GenerStage1DreamEff._run_shard,
                        self.backend,
                        self.model_path,
                        self.codedllm_root,
                        self.ppt,
                        shards[i],
                        self.ai_kwargs,
                        devices[i],
                        progress_queue,
                    )
                    future_sizes[fut] = len(shards[i])
                pbar = tqdm(total=total_tasks, desc="stage1 tasks")
                done = 0
                while done < total_tasks:
                    try:
                        step = progress_queue.get(timeout=0.2)
                        done += int(step)
                        pbar.update(int(step))
                    except queue.Empty:
                        pass
                    # Surface worker exception as soon as possible.
                    for fut in future_sizes:
                        if fut.done():
                            fut.result()
                pbar.close()
                manager.shutdown()
        self.write_manifest()


if __name__ == "__main__":
    fire.Fire(GenerStage1DreamEff)

