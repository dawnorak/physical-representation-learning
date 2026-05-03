#!/bin/bash
set -e

cd /home/ubuntu/physical-representation-learning

mkdir -p ./results

# Hardcoded defaults to use model assets from encoders/
ENCODER_CKPT="${1:-/home/ubuntu/physical-representation-learning/encoders/vjepa.pth}"
MODEL_CONFIG="/home/ubuntu/physical-representation-learning/encoders/config.yaml"
RESULTS_DIR="${2:-/home/ubuntu/physical-representation-learning/results/encoders_validation}"

if [ ! -f "$ENCODER_CKPT" ]; then
  echo "Encoder checkpoint not found: $ENCODER_CKPT"
  echo "Usage: $0 [optional_encoder_ckpt] [optional_results_dir]"
  exit 1
fi

if [ ! -f "$MODEL_CONFIG" ]; then
  echo "Model config not found: $MODEL_CONFIG"
  exit 1
fi

python -m physics_jepa.eval_frozen_regression \
  --dataset_name active_matter \
  --encoder_checkpoint "$ENCODER_CKPT" \
  --model_config "$MODEL_CONFIG" \
  --probe_type both \
  --results_dir "$RESULTS_DIR" \
  --num_workers 0
