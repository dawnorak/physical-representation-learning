#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Pass the pretrained checkpoint path as $1
CHECKPOINT_PATH="${1:?Pass the pretrained checkpoint path as the first argument}"
shift
python -m physics_jepa.finetune \
    configs/train_activematter_small_vjepa.yaml \
    --trained_model_path "$CHECKPOINT_PATH" \
    "$@"
