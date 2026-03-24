#!/bin/bash
# Local generation script for BIRD and Spider on GPUs 1,2
# Usage: bash do_question_generation_v23_local.sh
#
# Changes from v21:
#   - New models: gpt-oss-120b, Qwen3.5-122B-A10B-FP8, NVIDIA-Nemotron-3-Super-120B-A12B-FP8
#   - Two-phase generation: Phase 1 generates questions (no SQL), Phase 2 generates SQL via DIN-SQL
#   - Difficulty is re-assigned from SQL complexity (DifficultyConformance validator removed)

set -e

export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/taxonomy_env/lib:${LD_LIBRARY_PATH}"
export CPATH="${HOME}/miniconda3/envs/taxonomy_env/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${HOME}/miniconda3/envs/taxonomy_env/lib/python3.11/site-packages/nvidia/cublas/include:${HOME}/miniconda3/envs/taxonomy_env/targets/x86_64-linux/include:${CPATH}"

WORKDIR=/raid/homes/giovanni.sullutrone/taxonomy
MODELS_DIR=/raid/homes/giovanni.sullutrone/taxonomy/models

cd ${WORKDIR}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taxonomy_env

VERSION="v23"
SKIP_IF_EXISTS=true

MODEL_NAMES="${MODELS_DIR}/gpt-oss-120b ${MODELS_DIR}/Qwen3.5-122B-A10B-FP8 ${MODELS_DIR}/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"

# ── BIRD ──
DB_NAME="bird_dev"
DB_ROOT="/raid/homes/giovanni.sullutrone/datasets/bird_dev/dev_databases"
OUTPUT_PATH="results/question_generation/dev_generated_question_${VERSION}.json"
INTERMEDIATE_PATH="results/intermediate_results/dev_generated_question_${VERSION}"

if [ "$SKIP_IF_EXISTS" = true ] && [ -f "$OUTPUT_PATH" ]; then
    echo "Skipping BIRD - output file already exists: $OUTPUT_PATH"
else
    echo "Starting BIRD generation..."
    python do_question_generation.py \
        --db_name ${DB_NAME} \
        --db_root_path ${DB_ROOT} \
        --model_names ${MODEL_NAMES} \
        --n_samples 1 \
        --tensor_parallel_size 2 \
        --intermediate_results_folder ${INTERMEDIATE_PATH} \
        --output_path ${OUTPUT_PATH} \
        --styles formal \
        --resume \
        --verbose
    echo "BIRD generation complete."
fi

# ── Spider ──
DB_NAME="spider_test"
DB_ROOT="/raid/homes/giovanni.sullutrone/datasets/spider_test/test_databases"
OUTPUT_PATH="results/question_generation/spider_test_generated_question_${VERSION}.json"
INTERMEDIATE_PATH="results/intermediate_results/spider_test_generated_question_${VERSION}"

if [ "$SKIP_IF_EXISTS" = true ] && [ -f "$OUTPUT_PATH" ]; then
    echo "Skipping Spider - output file already exists: $OUTPUT_PATH"
else
    echo "Starting Spider generation..."
    python do_question_generation.py \
        --db_name ${DB_NAME} \
        --db_root_path ${DB_ROOT} \
        --model_names ${MODEL_NAMES} \
        --n_samples 1 \
        --tensor_parallel_size 2 \
        --intermediate_results_folder ${INTERMEDIATE_PATH} \
        --output_path ${OUTPUT_PATH} \
        --styles formal \
        --resume \
        --verbose
    echo "Spider generation complete."
fi

echo "All done."
