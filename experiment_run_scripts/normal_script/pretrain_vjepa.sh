#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/ubuntu/physical-representation-learning

mkdir -p ./checkpoints
mkdir -p ./logs

export THE_WELL_DATA_DIR=/home/ubuntu
export PYTHONPATH=/home/ubuntu/physical-representation-learning:$PYTHONPATH
export WANDB_API_KEY="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"
export WANDB_ENTITY="ssb10002-new-york-university"
export WANDB_PROJECT="physics-jepa"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

python -m physics_jepa.train_jepa \
  configs/train_activematter_small_vjepa.yaml \
  out_path=./checkpoints \
  dataset.resolution=128 \
  train.run_name=train_vjepa \
  train.num_epochs=20 \
  train.batch_size=32 \
  train.distributed=false \
  train.save_every=1 \
  +train.save_every_steps=2000 \
  train.report_every=50 \
  train.amp=false \
  train.lr=3e-4 \
  train.max_grad_norm=1.0
