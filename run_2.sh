#!/usr/bin/env bash
# Run from: cd /path/to/CWEval && conda activate cweval-dlm
#
# dream_multitask_ast: multitask Dream + ast_ex rank path + CWEval-aligned
# code fence extraction (generation_utils_ast_ex_cweval).
# Checkpoints: multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr ckpt 1200–1500.

# Podman is used as a Docker-compatible sandbox for running tests securely.
# The Docker Python SDK reads DOCKER_HOST to connect to Podman's socket.
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

# ── Stage 1: Generate ────────────────────────────────────────────────────────

export PYTHONPATH=.

CODEDLLM_ROOT="xxxx/codediffu/CodeDllm"

# eval_path suffix ``dream_mtast_cweval`` distinguishes outputs from ``dream_ast`` (-ast-inference) runs.
# Comment out any ``generate_stage1_dream`` / ``evaluate_stage2`` blocks you are not running.

python cweval/generate_stage1_dream.py gen \
  --backend dream_multitask_ast \
  --model_path "xxxx/codediffu/CodeDllm/projects/rl_dream_py_seceval/rl-seceval-frommultitask1300-margin/ckpt/round_14" \
  --codedllm_root "${CODEDLLM_ROOT}" \
  --eval_path evals/eval_multitask_ast_bigvul_cm_RLmarginckpt14_n4_4gpu-dream_mtast \
  --ppt direct \
  --num_proc 4 \
  --gpu_ids 0,1,2,3 \
  --langs py,c,cpp \
  --n 4 \
  --max_completion_tokens 512 \
  --steps 512 \
  --temperature 0.7 \
  --top_p 0.9 \
  --alg entropy \
  --alg_temp 0.1


# ── Stage 2: Evaluate ────────────────────────────────────────────────────────
# Set --eval_path to the same value as Stage 1 above.
# Use --docker True to run tests inside the Podman container (safe).

export PYTHONPATH=.
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

# checkpoint-1200
python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_RLmarginckpt14_n4_4gpu-dream_mtast \
  --num_proc 8 \
  --docker True \
  --k 4

###


