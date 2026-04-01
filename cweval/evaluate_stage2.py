"""
Stage-2 evaluation only for CWEval.

This script assumes generation has been completed (Stage-1) and only runs:
parse -> compile -> tests -> merge -> report.
"""

import json
import os
from typing import Dict

import fire
from natsort import natsorted

from cweval.evaluate import Evaler


class EvalerStage2(Evaler):
    def _read_manifest(self) -> Dict:
        manifest_path = os.path.join(self.eval_path, "generation_manifest.json")
        if not os.path.exists(manifest_path):
            return {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_generation_complete(self, strict: bool = True) -> bool:
        """
        Validate stage-1 outputs are complete before running evaluation.

        Check logic:
        1) Prefer generation_manifest.json if present.
        2) Fallback: ensure all generated_X directories have identical *_raw files
           relative to generated_0.
        """
        manifest = self._read_manifest()
        if manifest:
            incomplete = int(manifest.get("num_incomplete_cases", 0))
            if incomplete == 0:
                print("[stage2-check] manifest says generation complete.")
                return True
            msg = (
                f"[stage2-check] generation incomplete by manifest: "
                f"{incomplete} cases incomplete."
            )
            if strict:
                raise ValueError(msg)
            print(msg)
            return False

        if not self.generated_paths:
            msg = "[stage2-check] no generated_X directories found."
            if strict:
                raise ValueError(msg)
            print(msg)
            return False

        baseline = self.generated_paths[0]
        expected_rel_raw = set()
        for root, _, files in os.walk(baseline):
            if "__pycache__" in root:
                continue
            for file in natsorted(files):
                if "_raw." in file:
                    full = os.path.join(root, file)
                    expected_rel_raw.add(os.path.relpath(full, baseline))

        if not expected_rel_raw:
            msg = "[stage2-check] generated_0 has no *_raw files."
            if strict:
                raise ValueError(msg)
            print(msg)
            return False

        missing = []
        for generated_path in self.generated_paths:
            for rel in expected_rel_raw:
                cand = os.path.join(generated_path, rel)
                if not os.path.exists(cand):
                    missing.append(cand)

        if missing:
            msg = (
                f"[stage2-check] missing raw files: {len(missing)} "
                f"(example: {missing[0]})"
            )
            if strict:
                raise ValueError(msg)
            print(msg)
            return False

        print(
            f"[stage2-check] generation complete by scan: "
            f"{len(self.generated_paths)} generated dirs, {len(expected_rel_raw)} raw files each."
        )
        return True

    def stage2_parse(self) -> None:
        self.validate_generation_complete(strict=True)
        self.parse_generated()

    def stage2_compile(self) -> None:
        self.compile_parsed()

    def stage2_test(self, docker: bool = True) -> None:
        if isinstance(docker, str):
            docker = docker.lower() == "true"
        assert isinstance(docker, bool), f"{docker = }"
        if docker:
            self.run_tests_in_docker(prepare=False)
        else:
            self.run_tests()

    def stage2_merge(self) -> None:
        self._merge_results()

    def stage2_report(self) -> None:
        self.report_pass_at_k(mode="auto")

    def stage2_pipeline(self, docker: bool = True) -> None:
        self.validate_generation_complete(strict=True)
        self.parse_generated()
        self.compile_parsed()
        self.stage2_test(docker=docker)
        self._merge_results()
        self.report_pass_at_k(mode="auto")


if __name__ == "__main__":
    fire.Fire(EvalerStage2)

