#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/ubuntu/physical-representation-learning

export THE_WELL_DATA_DIR=/home/ubuntu
export PYTHONPATH=/home/ubuntu/physical-representation-learning:$PYTHONPATH

ENCODER_CKPT="${1:?Provide encoder checkpoint path}"
MODEL_CONFIG="${2:?Provide model config path}"
RESULTS_DIR="${3:-/home/ubuntu/physical-representation-learning/results/conv_large}"

python -m physics_jepa.eval_frozen_regression \
  --dataset_name active_matter \
  --encoder_checkpoint "$ENCODER_CKPT" \
  --model_config "$MODEL_CONFIG" \
  --probe_type both \
  --results_dir "$RESULTS_DIR" \
  --num_workers 0
