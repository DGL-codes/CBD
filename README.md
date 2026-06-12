# CBD

Code release for reproducing the CBD experiments. The repository contains source code, configuration files, and the unified reproduction entry point. It does not include datasets, model checkpoints, generated artifacts, result tables, audit logs, or paper-writing files.

## Installation

Create the Python environment from the provided environment file or install the package requirements manually.

```bash
conda env create -f environment.yaml
conda activate uld_exact_20260424
```

or

```bash
pip install -r requirements.txt
```

## External Resources

Datasets and models are expected to live outside the repository. Point the code to local copies through environment variables when needed.

Common variables:

```bash
export CBD_DATA_ROOT=/path/to/data/root
export PYTHON=/path/to/python
export REPRO_CONDA_ENV=uld_exact_20260424
export ASSIST_MODEL=/path/to/TinyLlama-1.1B-intermediate-step-1431k-3T
export TOFU_BASE_MODEL=/path/to/tofu_ft_llama2-7b
export BASE_MODEL=/path/to/zephyr-7b-beta
```

The code defaults to offline Hugging Face behavior. Set these variables if online loading is desired:

```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
```

Generated checkpoints, logs, bases, and evaluation outputs are written under `artifacts/`, which is intentionally ignored by git.

## Unified Entry Point

All public training, evaluation, and sweep commands go through:

```bash
python scripts/hf_forget_train.py repro --help
```

The entry point covers:

- Datasets: `ToFU01`, `ToFU05`, `ToFU10`, and `WMDP`
- White-box methods: `ga`, `ga+gd`, `ga+kl`, `dpo`, `dpo+gd`, `dpo+kl`, `npo`, `npo+gd`, `npo+kl`
- Gray-box methods: `uld`, `offset`
- Black-box method: `CSM-GE`
- Gray-box baseline: `GPM`
- Sweeps: `top_k`, basis retain size, basis forget size, LoRA rank, and forgetting steps

Use `--dry-run` to print the command that would be executed without starting training.

## Examples

White-box ToFU:

```bash
python scripts/hf_forget_train.py repro whitebox tofu dpo forget10 42 --gpus 0,1,2,3
```

Gray-box WMDP:

```bash
python scripts/hf_forget_train.py repro graybox wmdp uld 42 --split bio_cyber_chem --gpus 0,1,2,3
```

Black-box CSM-GE ToFU10:

```bash
python scripts/hf_forget_train.py repro blackbox tofu forget10 42 --top-k 192 --gpus 0
```

Black-box CSM-GE WMDP:

```bash
python scripts/hf_forget_train.py repro blackbox wmdp 42 --top-k 160 --gpus 0
```

GPM ToFU:

```bash
python scripts/hf_forget_train.py repro gpm tofu forget10 42 --gpus 0
```

Sweep examples:

```bash
python scripts/hf_forget_train.py repro sweep topk tofu10 --values 32,64,96,128,160,192 --gpus 0
python scripts/hf_forget_train.py repro sweep basis-retain wmdp --values 300,600,900,1200,1500 --gpus 0
python scripts/hf_forget_train.py repro sweep lora-r tofu10 --values 16,32,48,64,80 --gpus 0
python scripts/hf_forget_train.py repro sweep forget-steps tofu10 --values 60,120,180,240,300 --gpus 0
```

## Repository Contents

- `scripts/hf_forget_train.py`: unified public entry point
- `scripts/internal/`: internal implementations called by the unified entry point
- `scripts/eval_*.py`, `scripts/extract_*.py`, `scripts/select_*.py`: evaluation, basis extraction, and threshold utilities
- `configs/`: Hydra configurations used by the entry point
- `uld/`: core data, model, and trainer code

Internal scripts should not be used as public entry points directly.
