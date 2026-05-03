This is the official repository for the Deep Learning project code on representation learning for spatiotemporal physical systems.

**Note (model assets):** Due to Git LFS constraints, full model checkpoints and the matching Hydra config are not distributed through this repository. They are provided via the Drive link shared with the project report. Download the files you need and place them under `encoders/` in this repo (for example `encoders/ConvEncoder.pth` and `encoders/config.yaml`). For a sample end-to-end validation run, `run_saved_encoder_validation.sh` hardcoded `encoders/ConvEncoder.pth` with `encoders/config.yaml`. 

**OUR BEST MODEL IS V-JEPA, for the competition we would like you to place the model (in vjepa_checkpoints/VJepaVisionTransformer.pth and vjepa_checkpoints/config.yaml) into the encoders/ directory of this repo to get our performance.**

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

Install a PyTorch build compatible with your CUDA runtime if needed.

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

Each finetune script supports defaults derived from its filename (for example, `finetune_vjepa.sh` defaults to `encoders/vjepa.pth` and `encoders/config.yaml`). Copy the matching checkpoint and config from Drive into `encoders/`, or pass explicit paths as arguments to the script.

## Test Model Weights

This section is for validating shipped checkpoints from `encoders/`. For pretraining or finetuning other model variants, use the scripts under `experiment_run_scripts/` (`normal_script/` for local runs and `hpc_script/` for Slurm runs).

Use the dedicated script:

```bash
export THE_WELL_DATA_DIR=/path/to/well/root
chmod +x ./run_saved_encoder_validation.sh
./run_saved_encoder_validation.sh
```

This script hardcodes:

- checkpoint root: `./encoders/`
- config: `./encoders/config.yaml`
- default checkpoint: `ConvEncoder.pth` (conv-small-temporal sample shipped for the report)
- only `mkdir -p ./results` plus the Python validation command (no conda setup in the script; you must set `THE_WELL_DATA_DIR` and have dependencies installed as in Setup)
- optional: `MODEL_CONFIG=/path/to/config.yaml ./run_saved_encoder_validation.sh` to use a Drive-provided config without renaming it to `encoders/config.yaml`

You can optionally override checkpoint and results path (for example, after copying other checkpoints and configs from Drive into `encoders/`):

```bash
./run_saved_encoder_validation.sh \
  ./encoders/ConvEncoder.pth \
  ./results/encoders_validation
```

Underlying command used by the script:

```bash
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
