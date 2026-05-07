"""
Expected directory structure:

evals
├── eval_241110_014704
│   ├── generated_0
│   │   ├── core
│   │   │   ├── c
│   │   │   │   ├── compiled
│   │   │   │   │   └── cwe_022_0_c_task
│   │   │   │   ├── cwe_022_0_c_raw.c
│   │   │   │   ├── cwe_022_0_c_task.c    <--- Parse from _raw
│   │   │   │   ├── cwe_022_0_c_test.py    <--- Copy from benchmark
│   │   │   └── py
│   │   │       ├── cwe_020_0_raw.py
│   │   │       ├── cwe_020_0_task.py
│   │   │       ├── cwe_020_0_test.py
│   │   └── lang
│   │   └── res.json    <--- Run tests to get
│   └── generated_1
└── pytest.ini
"""

import datetime
import functools
import json
import math
import multiprocessing as mp
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
from natsort import natsorted
from p_tqdm import p_map

from cweval.commons import (
    BENCHMARK_DIR,
    COMPILE_DIR,
    LANGS,
    LANGS_COMPILE,
    PASS_AT_K_REPORT_FILENAME,
    append_pass_at_k_report_lines,
    compile_list,
    complete_code,
    get_code_from,
    pass_at_k,
    pass_at_k_report_filename,
    reset_pass_at_k_report_file,
    run_in_subprocess,
)
from cweval.run_tests import TestFileResult, run_tests
from cweval.sandbox import Container


class Evaler:

    entrypoint_anchor = 'BEGIN ENTRYPOINT'
    docker_user = 'ubuntu'
    repo_path_in_docker = f'/home/{docker_user}/CWEval'

    @staticmethod
    def _normalize_eval_langs(eval_langs: Any) -> List[str]:
        """Accept str, comma-str, or Fire-parsed tuple/list (e.g. ``('py','js')``)."""
        if eval_langs is None:
            return []
        if isinstance(eval_langs, (list, tuple)):
            return [str(x).strip() for x in eval_langs if str(x).strip()]
        s = str(eval_langs).strip()
        if not s:
            return []
        return [x.strip() for x in s.split(",") if x.strip()]

    def __init__(
        self,
        eval_path: str = '',
        num_proc: int = 8,
        eval_langs: Union[str, List[str], Tuple[str, ...], None] = None,
    ):
        if not eval_path:
            # find the latest one under './evals'
            evals_dir = 'evals'
            evals = natsorted(
                filter(
                    lambda x: os.path.isdir(os.path.join(evals_dir, x))
                    and x.startswith('eval_'),
                    os.listdir(evals_dir),
                )
            )
            eval_path = os.path.join(evals_dir, evals[-1])

        self.num_proc = num_proc
        self.eval_path = eval_path  # evals/eval_241110_014704
        # Restrict stage-2 parse/compile/test to these CWEval ``core/<lang>/`` trees
        # (comma-separated), e.g. ``py,js``. Empty = all languages (legacy behavior).
        self.eval_langs: List[str] = self._normalize_eval_langs(eval_langs)
        self.generated_paths = []
        for d in natsorted(os.listdir(self.eval_path)):
            if d.startswith('generated_'):
                self.generated_paths.append(os.path.join(self.eval_path, d))

        self.raw_files: List[str] = []
        self.task_files: List[str] = []  # parsed from raw_files

        # add all *_raw.* files to raw_files
        for generated_path in self.generated_paths:
            for root, dirs, files in os.walk(generated_path):
                if '__pycache__' in root:
                    continue
                for file in natsorted(files):
                    if '_raw.' in file:
                        self.raw_files.append(os.path.join(root, file))

        if self.eval_langs:
            before = len(self.raw_files)
            self.raw_files = [p for p in self.raw_files if self._under_eval_langs(p)]
            print(
                f'[eval_langs] filtered *_raw.* {before} -> {len(self.raw_files)} '
                f'({",".join(self.eval_langs)})',
                flush=True,
            )
        print(f'{len(self.raw_files) = }', flush=True)

    def _under_eval_langs(self, path: str) -> bool:
        if not self.eval_langs:
            return True
        norm = path.replace('\\', '/')
        return any(f'/core/{lg}/' in norm for lg in self.eval_langs)

    def _parse_raw_file(self, raw_file_path: str) -> str:
        # raw_code + lines after BEGIN ENTRYPOINT in ref_task_file
        # python cweval/evaluate.py _parse_raw_file --eval_path evals/eval_241110_014704
        with open(raw_file_path, 'r') as f:
            raw_str = f.read()

        raw_code = get_code_from(raw_str, only_first=True)
        if not raw_code:
            raw_code = raw_str

        # get the entrypoint from the corresponding task file
        for generated_path in self.generated_paths:
            if raw_file_path.startswith(generated_path.rstrip('/') + '/'):
                break
        rel_raw_file_path = os.path.relpath(raw_file_path, generated_path)
        ref_task_file_path = os.path.join(
            BENCHMARK_DIR, rel_raw_file_path.replace('_raw.', '_task.')
        )
        with open(ref_task_file_path, 'r') as ref_task_file:
            ref_task_code = ref_task_file.read()

        # TODO hack for python cases
        if self.entrypoint_anchor not in ref_task_code:
            return raw_code

        entrypoint_src_line = [
            line
            for line in ref_task_code.splitlines()
            if self.entrypoint_anchor in line
        ][0]
        entrypoint_code = ref_task_code.split(entrypoint_src_line)[1].strip()

        tot_code = f'{raw_code}\n\n{entrypoint_src_line}\n{entrypoint_code}\n'

        lang = os.path.splitext(raw_file_path)[1][1:]
        tot_code = complete_code(tot_code, lang)

        return tot_code

    def _fill_task_files(self) -> None:
        # fill the task_files with the task files
        if len(self.task_files) > 0:
            return
        for generated_path in self.generated_paths:
            for root, dirs, files in os.walk(generated_path):
                if '__pycache__' in root:
                    continue
                for file in natsorted(files):
                    if '_task.' in file:
                        full = os.path.join(root, file)
                        if self.eval_langs and not self._under_eval_langs(full):
                            continue
                        self.task_files.append(full)

    def _copy_test_files(self) -> None:
        # copy test files from benchmark to generated for testing
        self._fill_task_files()
        for task_file in self.task_files:
            test_file = (
                os.path.splitext(task_file.replace('_task.', '_test.'))[0] + '.py'
            )
            # evals/eval_241110_014704/generated_?/core/c/cwe_022_0_c_task.c -> evals/eval_241110_014704/generated_?
            for generated_path in self.generated_paths:
                if task_file.startswith(generated_path.rstrip('/') + '/'):
                    break
            rel_task_file_path = os.path.relpath(task_file, generated_path)
            ref_test_file_path = os.path.join(
                BENCHMARK_DIR,
                os.path.splitext(rel_task_file_path.replace('_task.', '_test.'))[0]
                + '.py',
            )
            # print(f'{ref_test_file_path} ==>> {test_file}')
            shutil.copy(ref_test_file_path, test_file)

    def _merge_results(self) -> None:
        # python cweval/evaluate.py _merge_results --eval_path evals/eval_241110_014704
        # merge the results from res.json files
        all_res: Dict[str, Dict[str, List[bool]]] = {}
        for generated_path in self.generated_paths:
            res_json_path = os.path.join(generated_path, 'res.json')
            with open(res_json_path, 'r') as f:
                res = json.load(f)
            for test_path, test_res in res.items():
                # evals/eval_241110_014704/generated_?/core/c/cwe_022_0_c_test.py -> evals/eval_241110_014704/generated_X/core/c/cwe_022_0_c_test.py
                generated_name = os.path.basename(generated_path)
                path_key = test_path.replace(generated_name, f'generated_X')
                all_res[path_key] = all_res.get(
                    path_key,
                    {
                        'functional': [],
                        'secure': [],
                        'func_secure': [],
                    },
                )
                all_res[path_key]['functional'].append(test_res['functional'])
                all_res[path_key]['secure'].append(test_res['secure'])
                all_res[path_key]['func_secure'].append(
                    test_res['functional'] and test_res['secure']
                )

        with open(os.path.join(self.eval_path, 'res_all.json'), 'w') as f:
            json.dump(all_res, f, indent=2)

    def _filename_to_lang(self, path: str) -> str:
        # path: evals/eval_241110_014704/generated_X/<...>/cwe_022_0_c_test.py -> c
        # evals/eval_241110_014704/generated_X/<...>/cwe_022_0_test.py -> py
        filename = os.path.splitext(os.path.basename(path))[0]
        lang = filename.split('_')[-2]
        if lang.isdigit():
            return 'py'
        return lang

    def report_pass_at_k(
        self,
        k: Optional[int] = None,
        lang: str = '',
        mode: str = 'auto',
        report_kind: str = '',
    ) -> Tuple[float, float, float] | None:
        rk = 'py_js' if report_kind == 'py_js' else 'standard'

        if mode == 'py_js_auto':
            # pass@k for k in 1..5; only core/py/, core/js/, and combined all(py+js).
            # Writes ``report_pass_at_{k}_py_js.txt`` (see ``commons.pass_at_k_report_filename``).
            if k is not None:
                ks = [k]
            else:
                ks = [1, 2, 3, 4, 5]
            for _k in ks:
                reset_pass_at_k_report_file(self.eval_path, k=_k, kind='py_js')
            for _lang in ('core/py/', 'core/js/', 'all(py+js)'):
                for _k in ks:
                    self.report_pass_at_k(_k, _lang, mode='', report_kind='py_js')
            for _k in ks:
                report_path = os.path.join(
                    self.eval_path, pass_at_k_report_filename(_k, kind='py_js')
                )
                print(f'[report] py+js summary saved to: {os.path.abspath(report_path)}')
            return

        if mode == 'auto':
            # If k is set, only that pass@k (still all language slices). If omitted, sweep 1/4/10/50.
            if k is not None:
                ks = [k]
            else:
                ks = [1, 4, 10, 50]
            # Reset one report file per k value so repeated calls don't accumulate stale lines.
            for _k in ks:
                reset_pass_at_k_report_file(self.eval_path, k=_k)
            for _lang in [f'core/{_l}/' for _l in LANGS] + [f'lang/c'] + ['']:
                # k must be <= num_samples in res_all (Stage-1 --n); larger k is skipped.
                for _k in ks:
                    self.report_pass_at_k(_k, _lang, mode='')
            for _k in ks:
                report_path = os.path.join(self.eval_path, pass_at_k_report_filename(_k))
                print(f'[report] Summary saved to: {os.path.abspath(report_path)}')
            return

        # Leaf: single (k, lang) block. Default k=1 when not specified (e.g. --mode _ only).
        effective_k = k if k is not None else 1

        all_res_json_path = os.path.join(self.eval_path, 'res_all.json')
        with open(all_res_json_path, 'r') as f:
            all_res = json.load(f)

        # filter by lang
        if lang == 'all(py+js)':
            all_res = {
                path: v
                for path, v in all_res.items()
                if 'core/py/' in path or 'core/js/' in path
            }
        elif lang:
            all_res = {path: v for path, v in all_res.items() if lang in path}

        if not all_res:
            return

        functional_patks: List[float] = []
        secure_patks: List[float] = []
        func_secure_patks: List[float] = []
        # secure_when_func_patks: List[float] = []
        for path, res in all_res.items():
            n_fun = len(res['functional'])
            if len(res['secure']) != n_fun or len(res['func_secure']) != n_fun:
                continue
            functional_patk = pass_at_k(
                n_fun,
                sum(res['functional']),
                effective_k,
            )
            # assert not any(not functional and secure for functional, secure in zip(res['functional'], res['secure'])), f'{path = } has a test case that is not functional but secure, which is impossible'
            secure_patk = pass_at_k(
                n_fun,
                sum(res['secure']),
                effective_k,
            )
            func_secure_patk = pass_at_k(
                n_fun,
                sum(res['func_secure']),
                effective_k,
            )

            # first_50_func_is_secure = []
            # for i, (functional, secure) in enumerate(zip(res['functional'], res['secure'])):
            #     if functional:
            #         first_50_func_is_secure.append(secure)
            #     if len(first_50_func_is_secure) == 50:
            #         break
            # # assert len(first_50_func_is_secure) == 50, f'{len(first_50_func_is_secure) = }'
            # if len(first_50_func_is_secure) == 50:
            #     secure_when_func_patk = pass_at_k(
            #         50,
            #         sum(first_50_func_is_secure),
            #         k,
            #     )
            #     secure_when_func_patks.append(secure_when_func_patk)

            if (
                math.isnan(functional_patk)
                or math.isnan(secure_patk)
                or math.isnan(func_secure_patk)
            ):
                # n < k for this problem: cannot estimate pass@k; do not count as 100%
                continue
            functional_patks.append(functional_patk)
            secure_patks.append(secure_patk)
            func_secure_patks.append(func_secure_patk)

        if not functional_patks:
            # n < k for every task in this slice (e.g. incomplete merge or k > --n).
            report_lines = [
                '=' * 16,
                f'pass@{effective_k}\t{lang or "all"}',
                f'func@{effective_k}\tN/A (no task has ≥{effective_k} samples)',
                f'sec@{effective_k}\tN/A (no task has ≥{effective_k} samples)',
                f'func-sec@{effective_k}\tN/A (no task has ≥{effective_k} samples)',
                '=' * 16,
            ]
            for line in report_lines:
                print(line)
            append_pass_at_k_report_lines(
                self.eval_path, report_lines, k=effective_k, kind=rk
            )
            return

        num_paths = len(functional_patks)
        functional_rate = sum(functional_patks) / num_paths * 100
        secure_rate = sum(secure_patks) / num_paths * 100
        func_secure_rate = sum(func_secure_patks) / num_paths * 100

        report_lines = [
            '=' * 16,
            f'pass@{effective_k}\t{lang or "all"}',
            f'func@{effective_k}\t{functional_rate:.2f}',
            f'sec@{effective_k}\t{secure_rate:.2f}',
            f'func-sec@{effective_k}\t{func_secure_rate:.2f}',
            '=' * 16,
        ]
        for line in report_lines:
            print(line)
        append_pass_at_k_report_lines(
            self.eval_path, report_lines, k=effective_k, kind=rk
        )

        return functional_rate, secure_rate, func_secure_rate

    def _parse_raw_write_task(self, raw_file: str) -> None:
        task_code = self._parse_raw_file(raw_file)
        task_file = raw_file.replace('_raw.', '_task.')
        with open(task_file, 'w') as f:
            f.write(task_code)

    def parse_generated(self) -> None:
        # python cweval/evaluate.py parse_generated --eval_path evals/eval_241110_014704
        # parse the raw_files to get the task_files
        if self.num_proc == 1:
            for raw_file in natsorted(self.raw_files):
                task_code = self._parse_raw_file(raw_file)
                task_file = raw_file.replace('_raw.', '_task.')
                self.task_files.append(task_file)
                with open(task_file, 'w') as f:
                    f.write(task_code)
        else:
            print(
                f'Parsing {len(self.raw_files)} files with {self.num_proc * 2} processes',
                flush=True,
            )
            p_map(
                self._parse_raw_write_task, self.raw_files, num_cpus=self.num_proc * 2
            )

    def compile_parsed(self) -> None:
        # python cweval/evaluate.py compile_parsed --eval_path evals/eval_241110_014704
        self._fill_task_files()
        # compile C
        to_compile_files = [
            task_file
            for task_file in self.task_files
            if any(task_file.endswith(f'.{lang}') for lang in LANGS_COMPILE)
        ]
        # {c_files_dir}/{COMPILE_DIR}/{name_of_c_file}
        compiled_files = [
            os.path.join(
                os.path.dirname(task_file),
                COMPILE_DIR,
                os.path.splitext(os.path.basename(task_file))[0],
            )
            for task_file in to_compile_files
        ]
        compile_list(
            to_compile_files, compiled_files, check=False, num_proc=self.num_proc
        )

    def run_tests(self) -> None:
        # python cweval/evaluate.py run_tests --eval_path evals/eval_241110_014704
        self._copy_test_files()
        all_gen_results = []
        el = self.eval_langs if self.eval_langs else None
        if self.num_proc == 1:
            for generated_path in self.generated_paths:
                # file_res_list = run_tests(generated_path)
                file_res_list = run_in_subprocess(
                    run_tests, generated_path, eval_langs=el
                )
                all_gen_results.append(file_res_list)
        else:
            mp.set_start_method('spawn', force=True)
            all_gen_results: List[TestFileResult] = []
            rt = functools.partial(run_tests, eval_langs=el)
            # fix mysterious hanging issue
            for i in range(math.ceil(len(self.generated_paths) / self.num_proc)):
                generated_paths_i = self.generated_paths[
                    i * self.num_proc : (i + 1) * self.num_proc
                ]
                assert len(generated_paths_i) <= self.num_proc
                with mp.Pool(self.num_proc, maxtasksperchild=1) as pool:
                    gen_results_i = pool.map(rt, generated_paths_i, chunksize=1)
                all_gen_results.extend(gen_results_i)
                print(f'Finished {i = } th batch', flush=True)

            # with mp.Pool(self.num_proc, maxtasksperchild=1) as pool:
            #     all_gen_results = pool.map(run_tests, self.generated_paths, chunksize=1)

        print(f'Finished running tests in {self.eval_path = }', flush=True)

        for file_res_list, generated_path in zip(all_gen_results, self.generated_paths):
            all_res = {
                file_res.file: {
                    'functional': file_res.functional,
                    'secure': file_res.secure,
                }
                for file_res in file_res_list
            }
            res_json_path = os.path.join(generated_path, 'res.json')
            with open(res_json_path, 'w') as f:
                json.dump(all_res, f, indent=4)

    def run_tests_in_docker(self, prepare: bool = True) -> None:
        if prepare:
            self.parse_generated()
            self.compile_parsed()
        print(f'Run docker', flush=True)
        timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        container = Container(
            image='co1lin/cweval',
            name=f'cweval_{timestamp}',
            user=self.docker_user,
        )
        # prepare the files in the container
        evals_path_in_docker = os.path.join(
            self.repo_path_in_docker, 'evals'
        )  # /home/ubuntu/CWEval/evals
        eval_path_in_docker = os.path.join(
            evals_path_in_docker, os.path.basename(self.eval_path)
        )  # /home/ubuntu/CWEval/evals/eval_241110_014704
        container.exec_cmd(
            f'''bash -c "
mkdir -p {evals_path_in_docker};
rm -rf {eval_path_in_docker}
"'''
        )
        container.copy_to(self.eval_path, eval_path_in_docker)
        # Fix ownership: files copied from host retain host uid; chown to container user
        container.exec_cmd(
            f'chown -R {self.docker_user}:{self.docker_user} {eval_path_in_docker}',
            user='root',
        )

        # Copy local cweval/ source into the container to ensure fixes in run_tests.py
        # etc. are picked up. Podman rootless bind-mounts suffer UID remapping that
        # makes mounted files unreadable by the container user; tarball-copy avoids this.
        _local_cweval_src = os.path.dirname(os.path.abspath(__file__))  # …/CWEval/cweval
        cweval_in_docker = os.path.join(self.repo_path_in_docker, 'cweval')
        container.copy_to(_local_cweval_src, cweval_in_docker)
        container.exec_cmd(
            f'chown -R {self.docker_user}:{self.docker_user} {cweval_in_docker}',
            user='root',
        )

        # Fix miniforge3 permissions so ubuntu can read activate regardless of Podman
        # user-namespace UID remapping (u+rX is insufficient; use a+rX).
        container.exec_cmd(
            f'chmod -R a+rX /home/{self.docker_user}/miniforge3',
            user='root',
        )

        log_path_in_docker = os.path.join(
            eval_path_in_docker, 'run_tests.log'
        )  # /home/ubuntu/CWEval/evals/eval_241110_014704/run_tests.log
        # run the tests — activate conda first, fall back to PATH python if unreadable
        miniforge_activate = f'/home/{self.docker_user}/miniforge3/bin/activate'
        eval_langs_cli = ''
        if self.eval_langs:
            eval_langs_cli = ' --eval_langs ' + ','.join(self.eval_langs)
        cmd = (
            f'bash -c \''
            f'set -e; '
            f'[ -r {miniforge_activate} ] && source {miniforge_activate}; '
            f'cd {self.repo_path_in_docker}; '
            f'source .env; '
            f'python cweval/evaluate.py run_tests '
            f'--eval_path {eval_path_in_docker} --num_proc {self.num_proc}'
            f'{eval_langs_cli} '
            f'2>&1 | tee {log_path_in_docker}'
            f'\''
        )
        exit_code, stdout, stderr = container.exec_cmd(cmd)
        assert exit_code == 0, f'{exit_code = }\nstdout:\n{stdout}\n\nstderr:\n{stderr}'
        # copy the log file and results
        log_path = os.path.join(
            self.eval_path, 'run_tests.log'
        )  # evals/eval_241110_014704/run_tests.log
        container.copy_from(log_path_in_docker, log_path)
        for generated_path in self.generated_paths:
            res_json_path = os.path.join(
                generated_path, 'res.json'
            )  # evals/eval_241110_014704/generated_X/res.json
            res_json_path_in_docker = os.path.join(
                eval_path_in_docker, os.path.relpath(res_json_path, self.eval_path)
            )  # /home/ubuntu/CWEval/evals/eval_241110_014704/generated_X/res.json
            container.copy_from(res_json_path_in_docker, res_json_path)

    def pipeline(self, docker: bool = True) -> None:
        self.parse_generated()
        self.compile_parsed()
        if isinstance(docker, str):
            docker = docker.lower() == 'true'
        assert isinstance(docker, bool), f'{docker = }'
        if docker:
            self.run_tests_in_docker(prepare=False)
        else:
            self.run_tests()
        self._merge_results()
        self.report_pass_at_k(mode='auto')


if __name__ == '__main__':
    fire.Fire(Evaler)
