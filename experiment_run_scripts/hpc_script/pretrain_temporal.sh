#!/bin/bash
#SBATCH --job-name=am-jepa-temporal
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=g2-standard-24
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=85G
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --output=/scratch/ssb10002/physical-representation-learning/logs/%x-%j.out
#SBATCH --error=/scratch/ssb10002/physical-representation-learning/logs/%x-%j.err
#SBATCH --requeue

set -e

source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate prl311

cd /scratch/ssb10002/physical-representation-learning
mkdir -p /scratch/ssb10002/checkpoints
mkdir -p /scratch/ssb10002/physical-representation-learning/logs

export THE_WELL_DATA_DIR=/scratch/ssb10002/data
export PYTHONPATH=/scratch/ssb10002/physical-representation-learning:$PYTHONPATH
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

torchrun --nproc_per_node=2 --standalone \
  -m physics_jepa.train_jepa \
  configs/train_activematter_small.yaml \
  out_path=/scratch/ssb10002/checkpoints \
  model=conv_small \
  model.physics_aware=false \
  train.temporal_masking=true \
  train.future_mask_frames=4 \
  train.num_epochs=20 \
  train.multi_scale_temporal_context=true \
  train.multi_scale_temporal_stride=4
  train.batch_size=4 \
  train.distributed=true \
  train.noise_std=0.0 \
  train.save_every=1 \
  train.run_name=pretrain_temporal \
  train.amp=true \
  +train.save_every_steps=2000 \
  +train.target_global_batch_size=64
