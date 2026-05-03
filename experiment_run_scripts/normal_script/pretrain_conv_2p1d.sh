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
  configs/train_activematter_small.yaml \
  model=conv_small \
  model.encoder_block_type=factorized_2plus1d \
  out_path=./checkpoints \
  train.num_epochs=25 \
  train.batch_size=8 \
  train.distributed=false \
  train.run_name=am_jepa_2p1d_small_1gpu_fast \
  dataset.resolution=128 \
  +train.save_every_steps=200
