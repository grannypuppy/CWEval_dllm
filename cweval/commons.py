import multiprocessing as mp
import os
import subprocess
import tempfile
from typing import Any, Callable, Iterable, List, Tuple

# Written next to res_all.json by evaluate.Evaler.report_pass_at_k.
# When k is known (single-k mode) the filename uses the concrete value,
# e.g. report_pass_at_1.txt / report_pass_at_4.txt.
# When sweeping multiple ks the fallback name is used.
PASS_AT_K_REPORT_FILENAME = 'report_pass_at_k.txt'


def pass_at_k_report_filename(k: 'int | None') -> str:
    """Return the report filename for a given k (or the sweep fallback)."""
    if k is not None:
        return f'report_pass_at_{k}.txt'
    return PASS_AT_K_REPORT_FILENAME


def reset_pass_at_k_report_file(eval_path: str, k: 'int | None' = None) -> str:
    """Truncate the pass@k summary file under eval_path (same dir as res_all.json)."""
    path = os.path.join(eval_path, pass_at_k_report_filename(k))
    with open(path, 'w', encoding='utf-8'):
        pass
    return path


def append_pass_at_k_report_lines(
    eval_path: str, lines: Iterable[str], k: 'int | None' = None
) -> None:
    """Append lines (same text as printed to stdout) to the report file."""
    path = os.path.join(eval_path, pass_at_k_report_filename(k))
    with open(path, 'a', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

import fire
import numpy as np
from natsort import natsorted

LANGS_COMPILE = ['c', 'cpp', 'go']
LANGS_RUN = ['py', 'js']
LANGS = LANGS_COMPILE + LANGS_RUN

COMPILE_DIR = 'compiled'
BENCHMARK_DIR = 'benchmark'


C_HEADERS = '''#include <archive_entry.h>
#include <jwt.h>
#include <errno.h>
#include <sys/wait.h>
#include <libxml/xpath.h>
#include <sqlite3.h>
#include <archive.h>
#include <ctype.h>
#include <libxml/parser.h>
#include <openssl/evp.h>
#include <openssl/des.h>
#include <openssl/aes.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <openssl/md5.h>
#include <stdlib.h>
#include <libgen.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <openssl/sha.h>
#include <regex.h>
#include <unistd.h>
#include <openssl/rand.h>
#include <archive.h>
#include <archive_entry.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <jwt.h>
#include <libgen.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <openssl/aes.h>
#include <openssl/des.h>
#include <openssl/evp.h>
#include <openssl/md5.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <regex.h>
#include <sqlite3.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
'''


CPP_HEADERS = '''#include <iostream>
#include <string>
#include <cstring>
#include <cctype>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <archive.h>
#include <archive_entry.h>
#include <memory>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <map>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <cstdio>
#include <openssl/aes.h>
#include <openssl/des.h>
#include <openssl/rand.h>
#include <tuple>
#include <stdexcept>
#include <jwt.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <sys/stat.h>
#include <utility>
#include <regex>
#include <sqlite3.h>
'''


PY_HEADERS = '''import hashlib
import html
import tarfile
from urllib.parse import urlparse
from typing import Tuple, List, Dict
import yaml
from Crypto.Cipher import AES, DES
import datetime
from tempfile import NamedTemporaryFile, TemporaryFile, mktemp
from Crypto.Cipher import AES
import jwt
import requests
import os
import json
from Crypto.PublicKey import DSA
from Crypto.PublicKey import RSA
from yaml import Loader
import argon2
import re
import subprocess
import sqlite3
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from lxml import etree
from Crypto.Cipher import AES
from Crypto.Cipher import AES, DES
from Crypto.PublicKey import DSA
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from lxml import etree
from tempfile import NamedTemporaryFile, TemporaryFile, mktemp
from urllib.parse import urlparse
from yaml import Loader
import argon2
import datetime
import hashlib
import html
import json
import jwt
import os
import re
import requests
import sqlite3
import subprocess
import tarfile
import yaml
'''


def complete_code(code: str, lang: str) -> str:
    if lang == 'c':
        return C_HEADERS + code
    elif lang == 'cpp':
        return CPP_HEADERS + code
    elif lang == 'py':
        return PY_HEADERS + code
    elif lang == 'go':
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(code)
            f.flush()
            filename = f.name
        cmd = f'goimports -w {filename} {filename}'
        exec_cmd_shell(cmd, check=False)
        with open(filename, 'r') as f:
            fixed_code = f.read()
        os.remove(filename)
        return fixed_code
    else:
        return code


def get_code_from(
    msg: str,
    only_last: bool = False,
    only_first: bool = False,
    add_new_line: bool = False,
) -> str:
    """
    Extract fenced code blocks from model output.

    Handles two failure modes common in diffusion language model outputs:

    1. Nested language fences inside docstrings (e.g. ```sql inside a C block):
       - The original logic treated any line starting with ``` as a closing fence,
         so ```sql in a comment would prematurely end extraction.
       - Fix: only treat a line as a closing fence when it starts with 3+ backticks
         followed ONLY by optional whitespace (no language specifier). Lines like
         ```sql, ```c, ```@param are treated as nested content, not closers.

    2. Closing ``` embedded mid-line (e.g. degenerate ``````the the the...):
       - The original logic never recognised these because it checked line.startswith.
       - Fix: after failing the line-start check, scan for the first ``` occurrence
         inside the line (idx > 0) and truncate the code there.

    3. Degenerate line-start fences like ``````words (DLLM often emits 4+ backticks
       then text on the same line). These must close the block; only ```lang (3 ticks
       + language) should stay as nested content.
       - Fix: if the leading backtick run length is >= 4 and there is non-whitespace
         after the run, treat as closing fence.
    """
    import re

    assert not (
        only_last and only_first
    ), '`only_last` and `only_first` cannot be both True'
    tail = '\n' if add_new_line else ''
    code_blocks: List[str] = []
    msg_lines = msg.splitlines()
    i_line = 0

    # A proper closing fence: 3+ backticks followed only by optional whitespace.
    # Any other line starting with ``` (e.g. ```sql, ```@param) is treated as content.
    _CLOSE_FENCE = re.compile(r'^`{3,}\s*$')

    def _leading_backtick_run(s: str) -> int:
        n = 0
        for ch in s:
            if ch == '`':
                n += 1
            else:
                break
        return n

    while i_line < len(msg_lines):
        line = msg_lines[i_line]
        if line.startswith('```'):
            code_lines = []
            i_line += 1
            while i_line < len(msg_lines):
                line = msg_lines[i_line]
                if line.startswith('```'):
                    if _CLOSE_FENCE.match(line):
                        # Plain ``` (only backticks + optional whitespace) — proper closing
                        break
                    run_len = _leading_backtick_run(line)
                    rest_after_run = line[run_len:]
                    if run_len >= 4 and rest_after_run.strip():
                        # e.g. ``````the the the — DLLM closing garbage on same line
                        break
                    # ```sql, ```@param, ```: etc. — nested or malformed fence.
                    # 3 ticks + language: keep as content; 4+ ticks with no trailing
                    # text falls through here (rare); plain-only long runs hit _CLOSE_FENCE.
                    code_lines.append(line)
                else:
                    # Check for ``` embedded in the middle of the line
                    # (degenerate diffusion output like ``````the the the...)
                    idx = line.find('```')
                    if idx > 0:
                        # Truncate code at the embedded backticks
                        code_lines.append(line[:idx].rstrip())
                        i_line += 1
                        break
                    code_lines.append(line)
                i_line += 1
            # end while for this code block
            code_blocks.append('\n'.join(code_lines) + tail)
            if only_first:
                return code_blocks[0]
        # end if for this code block
        i_line += 1
    # end while for all code blocks
    if only_last:
        return code_blocks[-1] if code_blocks else ''
    return '\n'.join(code_blocks)


def run_in_subprocess(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Run a function in a separate subprocess and return its result.

    Args:
        func: The function to run in the subprocess
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value from the function

    Example:
        def memory_intensive_function(x):
            # This will run in its own memory space
            large_list = [i for i in range(10**6)]
            return x * 2

        result = run_in_subprocess(memory_intensive_function, 5)
    """

    def wrapper(func, return_queue, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return_queue.put(result)
        except Exception as e:
            return_queue.put(e)

    # Create a queue to get the return value
    return_queue = mp.Queue()

    # Create and start the process
    process = mp.Process(
        target=wrapper, args=(func, return_queue) + args, kwargs=kwargs
    )
    process.start()

    # Wait for the process to complete
    process.join()

    # Get the result
    result = return_queue.get()

    # Check if the result is an exception
    if isinstance(result, Exception):
        raise result

    return result


def pass_at_k(n, c, k) -> float:
    """
    Unbiased pass@k estimate (single problem): n draws, c correct, evaluate pass@k.

    If ``n < k``, pass@k is not defined from the data (need at least k samples); returns
    NaN so callers can exclude this problem from pass@k averages.

    If ``n >= k`` and there are fewer than k incorrect samples (``n - c < k``), any
    k-subset must include at least one correct → estimate 1.0.
    """
    if n < k:
        return float('nan')
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def exec_cmd(
    cmd: List[str], check: bool = True, capture_output: bool = True
) -> Tuple[int, str, str]:
    assert isinstance(cmd, list)
    result = subprocess.run(cmd, capture_output=capture_output, text=True, check=check)
    return result.returncode, result.stdout, result.stderr


def exec_cmd_shell(
    cmd: str, check: bool = True, capture_output: bool = True
) -> Tuple[int, str, str]:
    assert isinstance(cmd, str)
    result = subprocess.run(
        cmd, capture_output=capture_output, text=True, check=check, shell=True
    )
    return result.returncode, result.stdout, result.stderr


def compile_c(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    lib_options = [
        '-lsqlite3',
        '-ljwt',
        # '-lcurl',
        '-lssl',
        '-lcrypto',
        '-larchive',
        '$(xml2-config --cflags --libs)',
    ]
    if 'lang/c' in src_path:
        exclude_patterns = [
            'lang/c/cwe_476_0_c_',
        ]
        if not any([pattern in src_path for pattern in exclude_patterns]):
            lib_options.append('-fsanitize=address')
    cmd = ['gcc', src_path, '-o', compiled_path] + lib_options
    cmd_str = ' '.join(cmd)
    returncode, stdout, stderr = exec_cmd_shell(cmd_str, check)
    if returncode != 0:
        print(f'Error compiling {src_path}:\n{stderr}', flush=True)
    return returncode, stdout, stderr


def compile_cpp(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    lib_options = [
        '-lsqlite3',
        '-ljwt',
        # '-lcurl',
        '-lssl',
        '-lcrypto',
        '-larchive',
        '$(xml2-config --cflags --libs)',
    ]
    if 'lang/cpp' in src_path:
        lib_options.append('-fsanitize=address')
    cmd = ['g++', src_path, '-o', compiled_path] + lib_options
    cmd_str = ' '.join(cmd)
    returncode, stdout, stderr = exec_cmd_shell(cmd_str, check)
    if returncode != 0:
        print(f'Error compiling {src_path}:\n{stderr}', flush=True)
    return returncode, stdout, stderr


def compile_go(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    cmd = ['go', 'build', '-o', compiled_path, src_path]
    cmd_str = ' '.join(cmd)
    returncode, stdout, stderr = exec_cmd_shell(cmd_str, check)
    if returncode != 0:
        print(f'Error compiling {src_path}:\n{stderr}', flush=True)
    return returncode, stdout, stderr


def compile_src(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    os.makedirs(os.path.dirname(compiled_path), exist_ok=True)
    lang = os.path.splitext(src_path)[1][1:]
    assert lang in LANGS_COMPILE, f'Unknown language for compile: {lang} for {src_path}'
    return {
        'c': compile_c,
        'cpp': compile_cpp,
        'go': compile_go,
    }[
        lang
    ](src_path, compiled_path, check)


def compile_list(
    src_path_list: List[str],
    compiled_path_list: List[str],
    check: bool = True,
    num_proc: int = 8,
) -> List[Tuple[int, str, str]]:
    # compiler sanity check
    cmd_checks = [
        'gcc --version',
        'g++ --version',
        'go version',
    ]
    for cmd_check in cmd_checks:
        returncode, stdout, stderr = exec_cmd_shell(cmd_check, check=True)

    assert len(src_path_list) == len(compiled_path_list)
    rets: List[Tuple[int, str, str]] = []
    if num_proc == 1:
        for src_path, compiled_path in zip(src_path_list, compiled_path_list):
            ret = compile_src(src_path, compiled_path, check)
            rets.append(ret)
    else:
        with mp.Pool(num_proc) as pool:
            rets = pool.starmap(
                compile_src,
                zip(src_path_list, compiled_path_list, [check] * len(src_path_list)),
            )
    return rets


def compile_all_in(
    path: str,
    check: bool = True,
    num_proc: int = 8,
) -> List[Tuple[int, str, str]]:
    src_path_list = []
    compiled_path_list = []
    if os.path.isfile(path):
        file_wo_ext, ext = os.path.splitext(path)
        if ext[1:] in LANGS_COMPILE:
            src_path_list.append(path)
            compiled_path = os.path.join(
                os.path.dirname(path), COMPILE_DIR, os.path.basename(file_wo_ext)
            )
            compiled_path_list.append(compiled_path)
    else:
        for root, _, files in os.walk(path):
            if '__pycache__' in root:
                continue
            for file in natsorted(files):
                file_wo_ext, ext = os.path.splitext(file)
                if ext[1:] in LANGS_COMPILE:
                    src_path = os.path.join(root, file)
                    compiled_path = os.path.join(root, COMPILE_DIR, file_wo_ext)
                    src_path_list.append(src_path)
                    compiled_path_list.append(compiled_path)

    return compile_list(src_path_list, compiled_path_list, check, num_proc)


if __name__ == '__main__':
    fire.Fire()
    # python cweval/commons.py compile_all_in --path benchmark
