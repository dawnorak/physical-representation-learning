#!/bin/bash
#SBATCH --job-name=am-ft-physicsaware
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=g2-standard-24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --output=/scratch/ssb10002/physical-representation-learning/logs/%x-%j.out
#SBATCH --error=/scratch/ssb10002/physical-representation-learning/logs/%x-%j.err
#SBATCH --requeue

set -e

source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate prl311

cd /scratch/ssb10002/physical-representation-learning
mkdir -p /scratch/ssb10002/physical-representation-learning/results

export THE_WELL_DATA_DIR=/scratch/ssb10002/data
export PYTHONPATH=/scratch/ssb10002/physical-representation-learning:$PYTHONPATH

MODEL_KEY="physicsaware"
DEFAULT_ENCODER="/scratch/ssb10002/physical-representation-learning/encoders/${MODEL_KEY}.pth"
ENCODER_CKPT="${1:-$DEFAULT_ENCODER}"
MODEL_CONFIG="${2:-/scratch/ssb10002/physical-representation-learning/encoders/config.yaml}"
RESULTS_DIR="${3:-/scratch/ssb10002/physical-representation-learning/results/physicsaware}"

if [ ! -f "$ENCODER_CKPT" ]; then
  echo "Encoder checkpoint not found: $ENCODER_CKPT"
  echo "Usage: sbatch $0 /path/to/encoder.pth [model_config] [results_dir]"
  exit 1
fi

python -m physics_jepa.eval_frozen_regression   --dataset_name active_matter   --encoder_checkpoint "$ENCODER_CKPT"   --model_config "$MODEL_CONFIG"   --probe_type both   --results_dir "$RESULTS_DIR"   --num_workers 0
