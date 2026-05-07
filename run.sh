#!/usr/bin/env bash
# Run from: cd /path/to/CWEval && conda activate cweval-dlm

# Podman is used as a Docker-compatible sandbox for running tests securely.
# The Docker Python SDK reads DOCKER_HOST to connect to Podman's socket.
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

# ── Stage 1: Generate ────────────────────────────────────────────────────────

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_multitask \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1200 \
#   --codedllm_root /research/jiamin0630/codediffu/CodeDllm \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1000_n4_4gpu_vulpatch \
#   --ppt vulpatch \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,cpp,c \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \
#   --test_limit 100

# python cweval/generate_stage1_dream.py gen \
#   --backend dream \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/sft-dream-base-sven-5e-7/checkpoint-800 \
#   --eval_path evals/eval_sft_base_ckpt800_bigvul_n4_4gpu \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \
#   --test_limit 100

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_ast \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/sft-dream-base-sven-5e-7/checkpoint-800 \
#   --eval_path evals/eval_sft_base_ckpt800_bigvul_n4_4gpu-ast-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1200 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1000_n4_4gpu_base-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_ast \
#   --model_path /research/jiamin0630/codediffu/dLLM-RL/local_models/dream-7b-base \
#   --eval_path evals/eval_dream-base_n4_4gpu-ast-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_ast \
#   --model_path /research/jiamin0630/codediffu/dLLM-RL/local_models/dream-7b-instruct \
#   --eval_path evals/eval_dream-instruct_n4_4gpu-ast-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1300 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_multitask \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1300 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

python cweval/generate_stage1_dream.py gen \
  --backend dream_ast \
  --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1300 \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu-ast-inference \
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
  --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_multitask \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1400 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1400 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu-base-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_ast \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1400 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu-ast-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# ###

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_multitask \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1500 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1500 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu-base-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# python cweval/generate_stage1_dream.py gen \
#   --backend dream_ast \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/multitask_stage2_bigvul_cm_1_1024_astdepth_dreambase_2e-6lr/checkpoint-1500 \
#   --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu-ast-inference \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

# ###

# python cweval/generate_stage1_dream.py gen \
#   --backend dream \
#   --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/sft-dream-base-sven-5e-7/checkpoint-1100 \
#   --eval_path evals/eval_sft_base_ckpt1100_bigvul_n4_4gpu \
#   --ppt direct \
#   --num_proc 4 \
#   --gpu_ids 0,1,2,3 \
#   --langs py,c,cpp \
#   --n 4 \
#   --max_completion_tokens 512 \
#   --steps 512 \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --alg entropy \
#   --alg_temp 0.1 \

python cweval/generate_stage1_dream.py gen \
  --backend dream_ast \
  --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/sft-dream-base-sven-5e-7/checkpoint-1100 \
  --eval_path evals/eval_sft_base_ckpt1100_bigvul_n4_4gpu-ast-inference \
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
  --alg_temp 0.1 \

###

python cweval/generate_stage1_dream.py gen \
  --backend dream \
  --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/sft-dream-base-sven-5e-7/checkpoint-1200 \
  --eval_path evals/eval_sft_base_ckpt1200_bigvul_n4_4gpu \
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
  --alg_temp 0.1 \

python cweval/generate_stage1_dream.py gen \
  --backend dream_ast \
  --model_path /research/jiamin0630/codediffu/CodeDllm/projects/sft_dream_py_ast/sft-dream-base-sven-5e-7/checkpoint-1200 \
  --eval_path evals/eval_sft_base_ckpt1200_bigvul_n4_4gpu-ast-inference \
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
  --alg_temp 0.1 \

# ── Stage 2: Evaluate ────────────────────────────────────────────────────────
# Set EVAL_PATH to the same --eval_path used in Stage 1 above.
# Use --docker True to run tests inside the Podman container (safe).

export PYTHONPATH=.
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_sft_base_ckpt1200_bigvul_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_sft_base_ckpt1200_bigvul_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 1

###


python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_sft_base_ckpt1100_bigvul_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_sft_base_ckpt1100_bigvul_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu-base-inference \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu-base-inference \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1500_n4_4gpu \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu-base-inference \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu-base-inference \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1400_n4_4gpu \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu-ast-inference \
  --num_proc 8 \
  --docker True \
  --k 1

###

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu \
  --num_proc 8 \
  --docker True \
  --k 4

python cweval/evaluate_stage2.py stage2_pipeline \
  --eval_path evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu \
  --num_proc 8 \
  --docker True \
  --k 1

###
# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_sft_base_ckpt800_bigvul_n4_4gpu-ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 4

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_sft_base_ckpt800_bigvul_n4_4gpu-ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 1

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_multitask_ast_bigvul_cm_ckpt1000_n4_4gpu_base-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 4

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_multitask_ast_bigvul_cm_ckpt1000_n4_4gpu_base-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 1


# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_dream-base_n4_4gpu-ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 4

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_dream-base_n4_4gpu-ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 1

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_dream-instruct_n4_4gpu-ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 4

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_dream-instruct_n4_4gpu-ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 1

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_multitask_ast_bigvul_cm_ckpt1000_n4_4gpu_ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 4

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_multitask_ast_bigvul_cm_ckpt1000_n4_4gpu_ast-inference \
#   --num_proc 8 \
#   --docker True \
#   --k 1
  
# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu \
#   --num_proc 8 \
#   --docker True \
#   --k 4

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path /research/jiamin0630/codediffu/CodeDllm/external/CWEval/evals/eval_multitask_ast_bigvul_cm_ckpt1300_n4_4gpu \
#   --num_proc 8 \
#   --docker True \
#   --k 1

# python cweval/evaluate_stage2.py stage2_pipeline \
#   --eval_path evals/eval_dream-7b-instruct-ckpt800_n4_4gpu \
#   --num_proc 8 \
#   --docker True