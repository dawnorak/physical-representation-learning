#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/ubuntu/physical-representation-learning

mkdir -p ./results

export THE_WELL_DATA_DIR=/home/ubuntu
export PYTHONPATH=/home/ubuntu/physical-representation-learning:$PYTHONPATH

MODEL_KEY="conv_small"
DEFAULT_ENCODER="/home/ubuntu/physical-representation-learning/encoders/${MODEL_KEY}.pth"
ENCODER_CKPT="${1:-$DEFAULT_ENCODER}"
MODEL_CONFIG="${2:-/home/ubuntu/physical-representation-learning/encoders/config.yaml}"
RESULTS_DIR="${3:-/home/ubuntu/physical-representation-learning/results/conv_small}"

if [ ! -f "$ENCODER_CKPT" ]; then
  echo "Encoder checkpoint not found: $ENCODER_CKPT"
  echo "Usage: $0 /path/to/encoder.pth [model_config] [results_dir]"
  exit 1
fi

python -m physics_jepa.eval_frozen_regression   --dataset_name active_matter   --encoder_checkpoint "$ENCODER_CKPT"   --model_config "$MODEL_CONFIG"   --probe_type both   --results_dir "$RESULTS_DIR"   --num_workers 0
