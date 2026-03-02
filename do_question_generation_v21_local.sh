#!/bin/bash
# Local generation script for BIRD and Spider on GPUs 2,3
# Usage: bash do_question_generation_v21_local.sh

set -e

export CUDA_VISIBLE_DEVICES=2,3

WORKDIR=/raid/homes/giovanni.sullutrone/taxonomy
MODELS_DIR=/raid/homes/giovanni.sullutrone/taxonomy/models

cd ${WORKDIR}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taxonomy_env

VERSION="v21"
SKIP_IF_EXISTS=true

MODEL_NAMES="${MODELS_DIR}/DeepSeek-R1-Distill-Llama-70B ${MODELS_DIR}/Qwen2.5-72B-Instruct ${MODELS_DIR}/Llama-3_3-Nemotron-Super-49B-v1"

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
        --n_samples 3 \
        --tensor_parallel_size 2 \
        --quantization fp8 \
        --intermediate_results_folder ${INTERMEDIATE_PATH} \
        --output_path ${OUTPUT_PATH} \
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
        --n_samples 3 \
        --tensor_parallel_size 2 \
        --quantization fp8 \
        --intermediate_results_folder ${INTERMEDIATE_PATH} \
        --output_path ${OUTPUT_PATH} \
        --verbose
    echo "Spider generation complete."
fi

echo "All done."
