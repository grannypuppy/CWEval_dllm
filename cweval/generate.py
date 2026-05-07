"""
Expected directory structure:

benchmark
├── core
│   ├── c
│   │   ├── cwe_022_0_c_task.c
│   └── py
│   |   ├── cwe_020_0_task.py
└── lang

evals
├── eval_241110_014704
│   ├── generated_0
│   │   ├── core
│   │   │   ├── c
│   │   │   │   ├── cwe_022_0_c_raw.c    <--- to generate
│   │   │   └── py
│   │   │       ├── cwe_020_0_raw.py
│   │   └── lang
│   └── generated_1
└── pytest.ini
"""

import datetime
import json
import os
import shutil
from typing import Any, Dict, List

import fire
from natsort import natsorted
from p_tqdm import p_map
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cweval.ai import AIAPI
from cweval.commons import BENCHMARK_DIR, LANGS
from cweval.ppt import make_prompt


_HF_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


class LocalHFAIAPI:
    def __init__(
        self,
        model_path: str,
        dtype: str = 'auto',
        device_map: str = 'auto',
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.dtype = dtype
        # 空串会导致 transformers 报错「found .」/ invalid device_map；与默认 `auto` 一致
        if not (device_map and str(device_map).strip()):
            device_map = 'auto'
        self.device_map = device_map
        self.req_kwargs = kwargs
        self.tokenizer, self.model = self._load_once()

    def _resolve_dtype(self):
        if self.dtype in ('auto', None):
            return 'auto'
        mapping = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if self.dtype not in mapping:
            raise ValueError(f'Unsupported dtype: {self.dtype}')
        return mapping[self.dtype]

    def _load_once(self):
        cache_key = f'{self.model_path}|{self.dtype}|{self.device_map}'
        if cache_key in _HF_MODEL_CACHE:
            return _HF_MODEL_CACHE[cache_key]['tokenizer'], _HF_MODEL_CACHE[cache_key]['model']

        torch_dtype = self._resolve_dtype()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
        )
        model.eval()
        _HF_MODEL_CACHE[cache_key] = {'tokenizer': tokenizer, 'model': model}
        return tokenizer, model

    def _build_input_text(self, messages: List[Dict[str, str]]) -> str:
        chat_template = getattr(self.tokenizer, 'chat_template', None)
        if hasattr(self.tokenizer, 'apply_chat_template') and chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return '\n\n'.join(m.get('content', '') for m in messages if m.get('content'))

    def send_message(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        all_kwargs = self.req_kwargs.copy()
        all_kwargs.update(kwargs)

        n_samples = int(all_kwargs.pop('n', 1))
        max_new_tokens = int(all_kwargs.pop('max_completion_tokens', 512))
        temperature = float(all_kwargs.pop('temperature', 0.0))
        top_p = all_kwargs.pop('top_p', None)
        use_cache = bool(all_kwargs.pop('use_cache', True))
        _ = all_kwargs.pop('lang', None)

        prompt_text = self._build_input_text(messages)
        encoded = self.tokenizer(prompt_text, return_tensors='pt')
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        input_len = encoded['input_ids'].shape[-1]

        do_sample = temperature > 0
        gen_common = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'pad_token_id': self.tokenizer.eos_token_id,
            'use_cache': use_cache,
        }
        if do_sample:
            gen_common['temperature'] = temperature
            if top_p is not None:
                gen_common['top_p'] = float(top_p)

        outputs: List[str] = []
        for _ in range(n_samples):
            with torch.no_grad():
                out_ids = self.model.generate(**encoded, **gen_common)
            new_ids = out_ids[0][input_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            outputs.append(text)
        return outputs


class Gener:

    begin_prompt_anchor = 'BEGIN PROMPT'
    begin_solution_anchor = 'BEGIN SOLUTION'

    def __init__(
        self,
        eval_path: str = '',
        backend: str = 'litellm',
        model: str = 'gpt-4o-mini-2024-07-18',
        model_path: str = '',
        ppt: str = 'direct',
        num_proc: int = 8,
        langs: List[str] = LANGS,
        exclude_path: List[str] = [],
        include_path: List[str] = [],
        # AI parameters
        n: int = 20,
        max_completion_tokens: int = 2048,
        temperature: float = 0.8,
        dtype: str = 'auto',
        device_map: str = 'auto',
        **kwargs,
    ):
        self.backend = backend
        self.model = model
        self.model_path = model_path
        self.ppt = ppt
        self.num_proc = num_proc
        self.langs = langs
        self.exclude_path = exclude_path
        self.include_path = include_path
        print(f'Using langs: {self.langs}')
        self.ai_kwargs = {
            'n': n,
            'max_completion_tokens': max_completion_tokens,
            'temperature': temperature,
            **kwargs,
        }
        if self.backend not in ('litellm', 'local_hf'):
            raise ValueError(f'Unsupported backend: {self.backend}')
        if self.backend == 'local_hf':
            if not self.model_path:
                raise ValueError('--model_path is required when --backend local_hf')
            # local_hf does not need model name for litellm
            self.model = self.model_path
            self.ai_kwargs['dtype'] = dtype
            self.ai_kwargs['device_map'] = device_map

        if not eval_path:
            self.eval_path = os.path.join(
                'evals', f'eval_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'
            )
        else:
            # check if eval_path exists
            if os.path.exists(eval_path):
                flag = (
                    input(f'{eval_path} already exists, overwrite? (y/n): ')
                    .strip()
                    .lower()
                )
                if flag != 'y':
                    print(f'Exiting...')
                    exit(0)

            self.eval_path = eval_path

        self.cases = self._get_cases()

    def _get_cases(self) -> Dict[str, Dict[str, str]]:
        cases: Dict[str, str] = {}
        for root, _, files in os.walk(BENCHMARK_DIR):
            if '__pycache__' in root:
                continue
            for file in natsorted(files):
                file_wo_ext, ext = os.path.splitext(file)
                task_file_path = os.path.join(root, file)
                lang = ext[1:]
                # filtering
                if not (ext and file_wo_ext.endswith('_task')):
                    continue
                if lang not in self.langs:
                    continue
                if any(exclude in task_file_path for exclude in self.exclude_path):
                    continue
                if self.include_path and not any(
                    include in task_file_path for include in self.include_path
                ):
                    continue
                # gather code prompt
                with open(task_file_path, 'r') as f:
                    task_code = f.read()
                begin_solution_line_src = ''
                for line in task_code.splitlines():
                    if self.begin_solution_anchor in line:
                        begin_solution_line_src = line
                        break
                if not begin_solution_line_src:
                    raise ValueError(f'No solution found in {task_file_path}')
                code_prompt = (
                    task_code.split(self.begin_prompt_anchor)[-1]
                    .split(begin_solution_line_src)[0]
                    .strip()
                )

                rel_task_file_path = os.path.relpath(task_file_path, BENCHMARK_DIR)
                gen_file_path_template = os.path.join(
                    self.eval_path,
                    'generated_{index}',
                    rel_task_file_path.replace('_task', '_raw'),
                )

                cases[task_file_path] = {
                    'task_file_path': task_file_path,
                    'code_prompt': code_prompt,
                    'lang': lang,
                    'out_path_template': gen_file_path_template,
                }

        return cases

    @staticmethod
    def _gen_case(
        backend: str,
        model: str,
        ppt: str,
        case: Dict[str, str],
        ai_kwargs: Dict[str, Any],
        rank: int,
    ) -> None:
        num_samples = ai_kwargs.get('n', 1)
        for i in range(num_samples):
            out_path = case['out_path_template'].format(index=i)
            if not os.path.exists(out_path):
                break
        else:
            print(
                f'{case["out_path_template"]} already completed, skipping', flush=True
            )
            return

        if backend == 'local_hf':
            aiapi = LocalHFAIAPI(model, **ai_kwargs)
        else:
            aiapi = AIAPI(model, **ai_kwargs)
        prompt = make_prompt(ppt)
        resps = prompt.req_ai(
            aiapi,
            case['lang'],
            case['code_prompt'],
            metadata={
                k: v for k, v in case.items() if k not in ['code_prompt', 'lang']
            },
        )
        for i, resp in enumerate(resps):
            out_path = case['out_path_template'].format(index=i)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(resp)

    def gen(self) -> None:
        p_map(
            self._gen_case,
            [self.backend] * len(self.cases),
            [self.model] * len(self.cases),
            [self.ppt] * len(self.cases),
            self.cases.values(),
            [self.ai_kwargs] * len(self.cases),
            range(len(self.cases)),  # workaround: index as rank
            num_cpus=self.num_proc,
        )


if __name__ == "__main__":
    fire.Fire(Gener)
