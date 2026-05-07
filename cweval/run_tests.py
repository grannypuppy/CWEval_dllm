import os
from dataclasses import dataclass, field
from typing import Dict, List

import pytest
from natsort import natsorted

CWD = os.getcwd()


@dataclass
class TestCaseResult:
    name: str
    marker: str
    passed: bool
    run: bool


@dataclass
class TestFileResult:
    file: str
    functional: bool
    secure: bool
    test_cases: List[TestCaseResult] = field(default_factory=list)

    def brief_str(self):
        return f'{__class__.__name__}(file=\'{self.file}\', functional={self.functional}, secure={self.secure})'


def _pytest_roots_for_generated(
    generated_path: str, eval_langs: List[str] | None
) -> List[str]:
    """If ``eval_langs`` is set (e.g. ``['py','js']``), only test under ``core/<lang>/``."""
    if not eval_langs:
        return [generated_path]
    roots = []
    for lg in eval_langs:
        sub = os.path.join(generated_path, "core", lg)
        if os.path.isdir(sub):
            roots.append(sub)
    return roots if roots else [generated_path]


def fill_missing_collection_failures(
    generated_path: str,
    file_results: List[TestFileResult],
    eval_langs: List[str] | None = None,
) -> List[TestFileResult]:
    """
    Pytest skips files that fail collection (e.g. ``SyntaxError`` when importing
    ``*_task`` from ``*_test.py``). Those files never enter ``TestResultCollector``,
    so they would be missing from ``res.json`` and break pass@k merge counts.

    Scan ``generated_path`` for every ``*_test.py``; any path not already present in
    ``file_results`` is treated as a failed run: ``functional=False``, ``secure=False``.
    """
    cwd = os.getcwd()
    seen_abs = {
        os.path.normpath(os.path.join(cwd, fr.file)) for fr in file_results
    }
    out = list(file_results)
    walk_roots = _pytest_roots_for_generated(generated_path, eval_langs)
    for walk_root in walk_roots:
        for root, _, files in os.walk(walk_root):
            if '__pycache__' in root:
                continue
            for name in natsorted(files):
                if not name.endswith('_test.py'):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, cwd)
                abs_path = os.path.normpath(os.path.join(cwd, rel))
                if abs_path in seen_abs:
                    continue
                out.append(
                    TestFileResult(
                        file=rel,
                        functional=False,
                        secure=False,
                        test_cases=[],
                    )
                )
                seen_abs.add(abs_path)
    return out


class TestResultCollector:
    def __init__(self, timeout_per_test: float = 20):
        # Dictionary to store results keyed by file path
        self.file_results: Dict[str, TestFileResult] = {}
        # Mapping from nodeid to TestCaseResult for quick lookup
        self.nodeid_to_test_case: Dict[str, TestCaseResult] = {}
        self.timeout_per_test = timeout_per_test

    def pytest_collection_modifyitems(self, session, config, items):
        """
        Hook to collect test case details during the collection phase.
        """
        for item in items:
            if item.get_closest_marker("functionality"):
                marker = "functionality"
            elif item.get_closest_marker("security"):
                marker = "security"
            else:
                continue
            # prevent hanging tests
            item.add_marker(pytest.mark.timeout(self.timeout_per_test, method="signal"))
            # nodeid example: 'tests/test_file1.py::test_case_a'
            nodeid = item.nodeid
            # Extract file path and test name
            file_path, test_name = nodeid.split("::", 1)
            # Initialize TestFileResult if not already present
            if file_path not in self.file_results:
                self.file_results[file_path] = TestFileResult(
                    file=os.path.relpath(item.path, CWD), functional=None, secure=None
                )

            # Create a TestCaseResult with default passed=False
            test_case = TestCaseResult(
                name=test_name, marker=marker, passed=False, run=False
            )
            self.file_results[file_path].test_cases.append(test_case)

            # Map nodeid to test_case_result for later reference
            self.nodeid_to_test_case[nodeid] = test_case

    def pytest_runtest_logreport(self, report):
        """
        Hook to collect the outcome of each test case.
        """
        if report.when == 'call':
            nodeid = report.nodeid
            test_case = self.nodeid_to_test_case.get(nodeid)
            if test_case is None:
                return
            test_case.run = True
            test_case.passed = report.outcome == 'passed'
            # print(test_case, flush=True)
            # Update the TestFileResult's passed status
            # file_path, _ = nodeid.split("::", 1)
            # if not test_case.passed:
            #     if test_case.marker == 'functionality':
            #         self.file_results[file_path].functional = False
            #     else:
            #         self.file_results[file_path].secure = False


def run_tests(
    test_path,
    timeout_per_test: float = 3,
    args: List[str] = ['-k', 'not _unsafe'],
    eval_langs: List[str] | None = None,
) -> List[TestFileResult]:
    roots = _pytest_roots_for_generated(test_path, eval_langs)
    print(f'Start running tests in {test_path = } roots={roots}', flush=True)
    result_collector = TestResultCollector(timeout_per_test=timeout_per_test)
    # temp fix:
    _os_exit = os._exit
    os._exit = lambda *args: None
    pytest.main(
        [*roots, '--tb=short', '--continue-on-collection-errors', *args],
        plugins=[result_collector],
    )
    os._exit = _os_exit
    print(f'[run_tests] Finished running tests in {test_path = }', flush=True)
    # compute file results
    for file_result in result_collector.file_results.values():
        # for test_case in file_result.test_cases:
        #     is_unsafe = '_unsafe' in test_case.name
        #     assert is_unsafe == (not test_case.run)
        file_result.functional = all(
            test_case.passed
            for test_case in file_result.test_cases
            if test_case.marker == 'functionality' and '_unsafe' not in test_case.name
        )
        file_result.secure = all(
            test_case.passed
            for test_case in file_result.test_cases
            if test_case.marker == 'security' and '_unsafe' not in test_case.name
        )
        # print(file_result.brief_str(), flush=True)

    results = list(result_collector.file_results.values())
    if os.path.isdir(test_path):
        results = fill_missing_collection_failures(
            test_path, results, eval_langs=eval_langs
        )
    return results


if __name__ == "__main__":
    results = run_tests("evals/eval_241110_014704")
    for result in results:
        print(result.brief_str())
