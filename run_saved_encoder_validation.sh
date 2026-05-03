#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$REPO_ROOT/results"

ENCODER_CKPT="${1:-$REPO_ROOT/encoders/ConvEncoder.pth}"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO_ROOT/encoders/config.yaml}"
RESULTS_DIR="${2:-$REPO_ROOT/results/encoders_validation}"

if [ ! -f "$ENCODER_CKPT" ]; then
  echo "Encoder checkpoint not found: $ENCODER_CKPT"
  echo "Usage: $0 [encoder_ckpt] [results_dir]"
  echo "Set MODEL_CONFIG to override config path (default: encoders/config.yaml)."
  exit 1
fi

if [ ! -f "$MODEL_CONFIG" ]; then
  echo "Model config not found: $MODEL_CONFIG"
  exit 1
fi

if [ -z "${THE_WELL_DATA_DIR:-}" ]; then
  echo "THE_WELL_DATA_DIR is not set. Point it at the Well dataset root (parent of data/train, data/valid, data/test)." >&2
  exit 1
fi

python -m physics_jepa.eval_frozen_regression \
  --dataset_name active_matter \
  --encoder_checkpoint "$ENCODER_CKPT" \
  --model_config "$MODEL_CONFIG" \
  --probe_type both \
  --results_dir "$RESULTS_DIR" \
  --num_workers 0
