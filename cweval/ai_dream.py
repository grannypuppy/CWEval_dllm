import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _default_codedllm_root() -> str:
    # Support both layouts:
    # 1) sibling: /.../codediffu/{CWEval,CodeDllm}
    # 2) submodule: /.../CodeDllm/external/CWEval
    here = Path(__file__).resolve()
    cweval_root = here.parent.parent
    candidates = [
        cweval_root.parent / "CodeDllm",  # sibling layout
        cweval_root.parent.parent,  # submodule layout (CodeDllm/external/CWEval)
    ]
    for cand in candidates:
        if (cand / "models").exists():
            return str(cand)
    # Fallback to sibling convention for backward compatibility.
    return str(candidates[0])


class DreamAPI:
    """
    Dream/Dream-multitask local inference adapter with AIAPI-like interface.

    send_message(messages, **kwargs) -> List[str]
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "dream",
        codedllm_root: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "bf16",
        use_rsp_prefix: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.codedllm_root = codedllm_root or _default_codedllm_root()
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_rsp_prefix = bool(use_rsp_prefix)
        self.seed = seed
        self.req_kwargs = kwargs
        self._model = None
        self._tokenizer = None

    def _lazy_init(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if self.codedllm_root not in sys.path:
            sys.path.insert(0, self.codedllm_root)

        import torch
        import types

        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))

        if self.backend == "dream_multitask":
            from models.dream_multitask import DreamModel
            from models.dream_multitask.tokenization_dream import DreamTokenizer
            from models.dream_multitask.generation_utils_ast_ex import (
                DreamGenerationMixin as AstExMixin,
            )

            self._model = DreamModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.torch_dtype == "bf16" else None,
                trust_remote_code=True,
            ).eval()
            # Ensure multitask uses ast_ex generation path.
            self._model.diffusion_generate = types.MethodType(
                AstExMixin.diffusion_generate, self._model
            )
            self._model._sample = types.MethodType(AstExMixin._sample, self._model)
            self._tokenizer = DreamTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
            )
        elif self.backend == "dream_multitask_ast":
            # Multitask Dream + ast_ex rank-weighted path, with CWEval-aligned
            # ``extract_code_from_output`` (``generation_utils_ast_ex_cweval``).
            from models.dream_multitask import DreamModel
            from models.dream_multitask.tokenization_dream import DreamTokenizer
            from models.dream_multitask.generation_utils_ast_ex_cweval import (
                DreamGenerationMixin as AstExCwevalMixin,
            )

            self._model = DreamModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.torch_dtype == "bf16" else None,
                trust_remote_code=True,
            ).eval()
            self._model.diffusion_generate = types.MethodType(
                AstExCwevalMixin.diffusion_generate, self._model
            )
            self._model._sample = types.MethodType(AstExCwevalMixin._sample, self._model)
            self._tokenizer = DreamTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
            )
        elif self.backend == "dream":
            from models import DreamModel, DreamTokenizer

            self._model = DreamModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.torch_dtype == "bf16" else None,
                trust_remote_code=True,
            ).eval()
            self._tokenizer = DreamTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
            )
        elif self.backend == "dream_ast":
            # Same base model as "dream", but monkey-patches the AST-guided
            # diffusion_generate / _sample from generation_utils_ast.py so that
            # every denoising step re-weights token confidence by AST node depth.
            from models import DreamModel, DreamTokenizer
            from models.dream.generation_utils_ast_cweval import (
                DreamGenerationMixin as AstMixin,
            )

            self._model = DreamModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.torch_dtype == "bf16" else None,
                trust_remote_code=True,
            ).eval()
            self._model.diffusion_generate = types.MethodType(
                AstMixin.diffusion_generate, self._model
            )
            self._model._sample = types.MethodType(AstMixin._sample, self._model)
            self._tokenizer = DreamTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
            )
        elif self.backend == "dream_bandit":
            # Same base Dream model, but monkey-patches Bandit-guided
            # diffusion path with CWEval-compatible extract_code_from_output.
            from models import DreamModel, DreamTokenizer
            from models.dream.generation_utils_bandit_cweval import (
                DreamGenerationMixin as BanditMixin,
            )

            self._model = DreamModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.torch_dtype == "bf16" else None,
                trust_remote_code=True,
            ).eval()
            self._model.diffusion_generate = types.MethodType(
                BanditMixin.diffusion_generate, self._model
            )
            self._model._sample = types.MethodType(BanditMixin._sample, self._model)
            self._tokenizer = DreamTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
            )
        elif self.backend == "dream_codeql":
            # Same base Dream model, but monkey-patches CodeQL-guided
            # diffusion path with CWEval-compatible fenced code extraction.
            from models import DreamModel, DreamTokenizer
            from models.dream.generation_utils_codeql_cweval import (
                DreamGenerationMixin as CodeQLMixin,
            )

            self._model = DreamModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.torch_dtype == "bf16" else None,
                trust_remote_code=True,
            ).eval()
            self._model.diffusion_generate = types.MethodType(
                CodeQLMixin.diffusion_generate, self._model
            )
            self._model._sample = types.MethodType(CodeQLMixin._sample, self._model)
            self._tokenizer = DreamTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        if self.device.startswith("cuda") and torch.cuda.is_available():
            self._model = self._model.to(self.device)
        else:
            self._model = self._model.to("cpu")

    def send_message(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        self._lazy_init()
        import torch

        all_kwargs = self.req_kwargs.copy()
        all_kwargs.update(kwargs)

        n = int(all_kwargs.pop("n", 1))
        max_completion_tokens = int(all_kwargs.pop("max_completion_tokens", 1024))
        temperature = float(all_kwargs.pop("temperature", 0.0))
        top_p = float(all_kwargs.pop("top_p", 0.95))
        top_k = all_kwargs.pop("top_k", None)
        steps = int(all_kwargs.pop("steps", 256))
        alg = str(all_kwargs.pop("alg", "entropy"))
        alg_temp = float(all_kwargs.pop("alg_temp", 0.1))
        threshold = all_kwargs.pop("threshold", None)
        shrink = int(all_kwargs.pop("shrink", 8))
        codeql_penalty_ratio = float(all_kwargs.pop("codeql_penalty_ratio", 0.5))
        codeql_timeout_sec = float(all_kwargs.pop("codeql_timeout_sec", 10.0))
        codeql_bin = all_kwargs.pop("codeql_bin", None)
        codeql_query_path = all_kwargs.pop("codeql_query_path", None)
        # lang is forwarded by ppt.req_ai; dream_ast/dream_bandit use it in generation utils.
        lang = str(all_kwargs.pop("lang", "python"))
        # Ignored for backward compatibility (block/cached generation path deprecated).
        for _k in ("use_cache", "dual_cache", "block_length"):
            all_kwargs.pop(_k, None)

        prompt_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            add_special_tokens=False,
        )
        if self.use_rsp_prefix:
            rsp_prefix = "Here is the code:\n```\n"
            prefix_ids = self._tokenizer(
                rsp_prefix, add_special_tokens=False
            ).input_ids
            prompt_ids = prompt_ids + prefix_ids

        input_ids_single = torch.tensor([prompt_ids], dtype=torch.long).to(
            self._model.device
        )
        attention_mask_single = torch.ones_like(input_ids_single)

        gen_kwargs = {
            "max_new_tokens": max_completion_tokens,
            "output_history": True,
            "return_dict_in_generate": True,
            "steps": steps,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "alg": alg,
            "alg_temp": alg_temp,
            "threshold": threshold,
            "tokenizer": self._tokenizer,
        }
        if self.backend == "dream_codeql":
            gen_kwargs["shrink"] = shrink
            gen_kwargs["codeql_penalty_ratio"] = codeql_penalty_ratio
            gen_kwargs["codeql_timeout_sec"] = codeql_timeout_sec
            if codeql_bin:
                gen_kwargs["codeql_bin"] = codeql_bin
            if codeql_query_path:
                gen_kwargs["codeql_query_path"] = codeql_query_path
        # Forward ``lang`` when backend generation utils accept language-aware extraction/parsing.
        if self.backend in ("dream_ast", "dream_multitask_ast", "dream_bandit", "dream_codeql"):
            gen_kwargs["lang"] = lang

        outputs: List[str] = []
        # Align behavior with CodeDllm benchmark scripts:
        # confidence_threshold is generated one-by-one, others batched.
        if alg == "confidence_threshold":
            batch_input_ids = input_ids_single
            batch_attention_mask = attention_mask_single
            repeats = n
        else:
            # Memory-safe chunking for long diffusion; avoid large in-batch repeat(n, 1)
            # that can OOM on multitask/base dream models.
            MEMORY_SAFE_MAX_REPEAT = 4
            steps_cap = 512
            if steps >= steps_cap and n > MEMORY_SAFE_MAX_REPEAT:
                chunk_repeat = MEMORY_SAFE_MAX_REPEAT
                repeats = (n + chunk_repeat - 1) // chunk_repeat
            else:
                chunk_repeat = n
                repeats = 1
            batch_input_ids = input_ids_single.repeat(chunk_repeat, 1)
            batch_attention_mask = attention_mask_single.repeat(chunk_repeat, 1)

        for _ in range(repeats):
            with torch.no_grad():
                output = self._model.diffusion_generate(
                    batch_input_ids, attention_mask=batch_attention_mask, **gen_kwargs
                )
            sequences = output.sequences if hasattr(output, "sequences") else output
            generated_texts = self._tokenizer.batch_decode(
                [seq[len(prompt_ids):] for seq in sequences],
                skip_special_tokens=True,
            )
            outputs.extend(generated_texts)

        return outputs[:n]

