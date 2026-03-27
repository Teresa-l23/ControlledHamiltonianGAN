#!/bin/bash
set -e

BASE_DIR="/home/jiayinliu/projects/SympNet"

cd "${BASE_DIR}"
python scripts/eval_all_systems.py \
  --base-dir "${BASE_DIR}" \
  --config-name configuration_60.ini \
  --output-dir "${BASE_DIR}/eval_results_all_systems" \
  --num-trajectory-samples 100 \
  --latent-batch-size 1024 \
  --eval-batch-size 64