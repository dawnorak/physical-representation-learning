This is the official repository for the Deep Learning project code on representation learning for spatiotemporal physical systems.

## Setup

### 1) Create and activate a Python environment

Use Conda (recommended):

```bash
conda create -n prl310 python=3.10 -y
conda activate prl310
```

### 2) Install dependencies

From the repository root:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA runtime if needed. Then install the package in editable mode:

```bash
pip install -e .
```

### 3) Set required environment variables

All training and validation scripts require dataset and import paths:

```bash
export THE_WELL_DATA_DIR=/path/to/well/root
export PYTHONPATH=$(pwd):$PYTHONPATH
```

For W\&B logging in online mode:

```bash
export WANDB_API_KEY="your_key_here"
export WANDB_ENTITY="your_entity"
export WANDB_PROJECT="physics-jepa"
```

## Running scripts

Experiment scripts are organized under:

- `experiment_run_scripts/normal_script` for interactive/single-node runs
- `experiment_run_scripts/hpc_script` for Slurm batch runs

### Pretraining

Use model-specific pretraining scripts, for example:

```bash
bash experiment_run_scripts/normal_script/pretrain_vjepa.sh
bash experiment_run_scripts/normal_script/pretrain_conv_large.sh
```

HPC equivalents can be submitted with `sbatch`.

## Validation from saved encoders

Saved encoder checkpoints can be validated with the finetune/validation scripts in `experiment_run_scripts`.

Each finetune script now supports direct defaults from `encoders/`:

- default checkpoint: `encoders/<model_name>.pth`
- default config: `encoders/config.yaml`

So if your files exist in `encoders/`, you can run validation with no extra arguments.

## Test Model Weights

This section is for validating shipped checkpoints from `encoders/`. For pretraining or finetuning other model variants, use the scripts under `experiment_run_scripts/` (`normal_script/` for local runs and `hpc_script/` for Slurm runs).

Use the dedicated script:

```bash
chmod +x ./run_saved_encoder_validation.sh
./run_saved_encoder_validation.sh
```

This script hardcodes:

- checkpoint root: `./encoders/`
- config: `./encoders/config.yaml`
- default checkpoint: `vjepa.pth`
- only `mkdir -p ./results` plus the Python validation command (no conda/env setup)

You can optionally override checkpoint and results path:

```bash
./run_saved_encoder_validation.sh \
  ./encoders/conv_2p1d.pth \
  ./results/encoders_conv_2p1d
```

Underlying command used by the script:

python -m physics_jepa.eval_frozen_regression \
  --dataset_name active_matter \
  --encoder_checkpoint "$ENCODER_CKPT" \
  --model_config "$MODEL_CONFIG" \
  --probe_type both \
  --results_dir "$RESULTS_DIR" \
  --num_workers 0
```

### Normal run examples

```bash
bash experiment_run_scripts/normal_script/finetune_vjepa.sh
bash experiment_run_scripts/normal_script/finetune_conv_small.sh
bash experiment_run_scripts/normal_script/finetune_conv_2p1d.sh
```

Optional explicit form:

```bash
bash experiment_run_scripts/normal_script/finetune_vjepa.sh \
  /path/to/encoder.pth \
  /path/to/config.yaml \
  /path/to/results_dir
```

### HPC run examples

```bash
sbatch experiment_run_scripts/hpc_script/finetune_vjepa.sh
sbatch experiment_run_scripts/hpc_script/finetune_conv_large.sh
```

Optional explicit form:

```bash
sbatch experiment_run_scripts/hpc_script/finetune_vjepa.sh \
  /scratch/.../encoder.pth \
  /scratch/.../config.yaml \
  /scratch/.../results_dir
```

## Notes

- Finetune scripts run `physics_jepa.eval_frozen_regression` with `--probe_type both` for validation.
- Results are written to per-model subdirectories under `results/` by default.
- If the checkpoint path does not exist, the script prints usage and exits with an error.
