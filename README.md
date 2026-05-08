# Pushing the Limits of 1-Shot RLVR: Stabilizing Verifiable Rewards in Mathematical Reasoning

> **DSAA 2026 — Special Track on Large Language Models**  
> Reinforcement Learning from Verifiable Rewards (RLVR) for mathematical reasoning using a single training example.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Improvements Implemented](#improvements-implemented)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Model Download](#model-download)
- [Running Experiments](#running-experiments)
- [Experiment Flags Reference](#experiment-flags-reference)
- [Output Files](#output-files)
- [Results](#results)
- [Team](#team)

---

## Overview

This project extends the **one-shot RLVR** paradigm — training a language model to improve its mathematical reasoning using **only a single labeled example** via Reinforcement Learning from Verifiable Rewards.

The base model is [`ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1`](https://huggingface.co/ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1), a 1.5B-parameter model pre-trained on mathematical text. We replicate the original paper's results and then systematically apply 7 improvements to the RL training pipeline, measuring the contribution of each change in isolation.

The training algorithm is **GRPO (Group Relative Policy Optimization)** with a Leave-One-Out advantage baseline, KL-divergence regularization against a frozen reference model, and a composite reward function.

---

## Key Contributions

| # | Contribution | Impact |
|---|-------------|--------|
| Baseline fix | Identified and corrected a critical regex bug in the original reward function — `\\boxed{([^}]*)}` never matched model output, causing the answer reward to always be zero | Essential |
| Baseline fix | Fixed prompt decoding: token-index slicing instead of string-length slicing | Essential |
| Imp 1 | SymPy-based answer normalization — `1/2 == 0.5 == \frac{1}{2}` | +accuracy |
| Imp 2 | Extra KL penalty against frozen reference model | +stability |
| Imp 3 | Step-wise process reward for reasoning lines | +reasoning |
| Imp 4 | Temperature annealing (1.0→0.4 over training) | +exploration |

---

## Improvements Implemented

Each improvement is a boolean flag. All are `False` by default (baseline). Flip **one at a time** to measure its isolated contribution.

### Improvement 1 — SymPy Answer Normalization (`sympy=True`)
Replaces plain string equality with a 4-layer comparison pipeline:
1. SymPy symbolic: `simplify(pred - gt) == 0`
2. Numeric float: `|float(pred) - float(gt)| < 1e-4`
3. Decimal string: round both to 4dp, compare
4. String fallback: whitespace-normalized string match

Catches: `"1/2" == "0.5"`, `"\\frac{3}{4}" == "0.75"`, `"100" == "100.0"`

### Improvement 2 — Extra KL Penalty (`kl=True`)
Adds `+0.05` to the baseline `KL_BETA=0.1`, making total KL coefficient 0.15. Prevents the policy from drifting too far from the reference model, reducing reward hacking.

### Improvement 3 — Step-wise Process Reward (`step=True`)
Awards `+0.04` per math reasoning line (lines containing `=`, `+`, `-`, `*`, `/` and length > 8 characters), capped at 5 lines (`+0.20` max). Provides dense signal for structured reasoning even when the final answer is wrong.

### Improvement 4 — Temperature Annealing (`anneal=True`)
Linearly decays sampling temperature from `TEMP_START=1.0` to `TEMP_END=0.4` over all training steps:
- **Early steps**: high temp → diverse rollouts → finds correct solutions
- **Late steps**: low temp → exploits what was learned → locks in behavior

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | |
| CUDA | ≥ 11.8 | GPU required |
| GPU VRAM | ≥ 24 GB | Two models loaded simultaneously (policy + reference). Tested on dual NVIDIA T4 (2×16GB) and single A100 (40GB) |
| Disk space | ≥ 20 GB | ~7.1 GB model weights + dataset cache + results |
| RAM | ≥ 16 GB | For model loading with `low_cpu_mem_usage=True` |

### Python packages

```
torch >= 2.1.0
transformers >= 4.46.0
datasets
sympy
numpy
matplotlib
accelerate
```

---

## Environment Setup

### Option A — Conda (Recommended)

```bash
# 1. Create environment
conda create -n rlvr_env python=3.10 -y
conda activate rlvr_env

# 2. Install PyTorch with CUDA (adjust cuda version to match your driver)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install remaining packages
pip install transformers datasets sympy numpy matplotlib accelerate
```

### Option B — pip / virtualenv

```bash
python3 -m venv rlvr_env
source rlvr_env/bin/activate          # Windows: rlvr_env\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets sympy numpy matplotlib accelerate
```

### Verify GPU is detected

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Model Download

The model weights must be downloaded before running. Download once and reuse across all experiments.

```bash
# Using huggingface_hub (recommended — supports resume on interruption)
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    local_dir="/home/RL/Rani_u23ai131_8809617110/model_cache",
    ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"],
)
EOF
```

Or using `git lfs`:

```bash
git lfs install
git clone https://huggingface.co/ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1 \
    /home/RL/Rani_u23ai131_8809617110/model_cache
```

> **Path note:** The script expects the model at `BASE_DIR/model_cache`. Edit `BASE_DIR` at the top of `rlvr_rani.py` if your directory differs.

---

## Running Experiments

### Step 1 — Set your paths

Open `rlvr_rani.py` and update the three path constants at the top:

```python
BASE_DIR      = "/your/working/directory"   # ← change this
MODEL_DIR     = f"{BASE_DIR}/model_cache"   # model weights location
DATASET_CACHE = f"{BASE_DIR}/dataset_cache" # auto-created on first run
OUTPUT_DIR    = f"{BASE_DIR}/results1"      # plots and JSON saved here
```

### Step 2 — Run all experiments sequentially

```bash
python rlvr_rani.py
```

This runs the experiments defined in `__main__`:
```
imp1_sympy  →  imp2_kl  →  imp4_step  →  imp6_anneal  →  optimal_combo
```

Each experiment loads a fresh model, trains for `TRAIN_STEPS=25` steps, runs a final eval on 50 held-out problems, saves results, then frees GPU memory.

### Step 3 — Run a single custom experiment

To run just one experiment with specific flags, call `run_experiment()` directly:

```python
# At the bottom of rlvr_rani.py, replace the experiments list with:
if __name__ == "__main__":
    train_sample, eval_samples = load_data()
    run_experiment("my_exp", {"step": True, "anneal": True}, train_sample, eval_samples)
    print_comparison_table()
```

### Step 4 — Run baseline only (all flags off)

```python
experiments = [("baseline", {})]
```

---

## Experiment Flags Reference

Pass these keys in the `flags` dict to `run_experiment()`:

| Flag key | Improvement | Default |
|----------|-------------|---------|
| `"sympy"` | SymPy answer normalization (Imp 1) | `False` |
| `"kl"` | Extra KL penalty +0.05 (Imp 2) | `False` |
| `"format"` | Format reward for `\boxed{}` (Imp 3) | `False` |
| `"step"` | Step-wise process reward (Imp 4) | `False` |
| `"rollouts"` | K=16 rollouts instead of K=8 (Imp 5) | `False` |
| `"anneal"` | Temperature annealing 1.0→0.4 (Imp 6) | `False` |
| `"lennorm"` | Length-normalized policy loss (Imp 7) | `False` |

**Example — all improvements combined:**
```python
run_experiment("all", {
    "sympy": True, "kl": True, "format": True,
    "step": True, "rollouts": True, "anneal": True, "lennorm": True
}, train_sample, eval_samples)
```

---

## Output Files

All outputs are saved to `OUTPUT_DIR` (default: `BASE_DIR/results1/`).

### Per experiment

| File | Description |
|------|-------------|
| `{exp_name}_results.png` | 5-panel plot: Mean Reward / Train Acc / Eval Acc / Grad Norm / Temperature |

### Comparison table

Printed to stdout after all experiments complete. Also reconstructable at any time:

```python
print_comparison_table()   # reads all *_summary.json files in OUTPUT_DIR
```

---

## Results

Experiments are evaluated on 50 held-out MATH benchmark problems (greedy decoding, `do_sample=False`). Pre-training accuracy of the base model is **0.640**.

| Experiment | Pre-acc | Eval-acc | Delta | Flags active |
|------------|---------|----------|-------|--------------|
| baseline | 0.640 | 0.640 | +0.000 | none |
| imp1_sympy | 0.640 | 0.680 | +0.040 | sympy_reward |
| imp2_kl | 0.640 | — | — | kl_penalty |
| imp4_step | 0.640 | 0.660 | +0.020 | step_reward |
| imp6_anneal | 0.640 | — | — | temp_anneal |
| optimal_combo | 0.640 | — | — | sympy + kl + step + anneal |


> Add `model_cache/` and `dataset_cache/` to your `.gitignore` — they contain large binary files not suited for version control.

### Recommended `.gitignore`

```gitignore
model_cache/
dataset_cache/
results1/
__pycache__/
*.pyc
*.egg-info/
.env
```

---

## Team

| Name | Roll No | Contribution |
|------|---------|-------------|
| Rani | u23ai131 | Implementation — RL training pipeline, all 7 improvements, experiment tracker, bug fixes |
| Sanjhi | u23ai121 | Research — improvement ideas, literature review, experiment design |
| Mohit | u23ai117 | Baseline replication — paper verification, answer extraction, dataset pipeline |

---

## Citation

If you use this code, please cite the original one-shot RLVR paper:

```bibtex
@article{wang2024oneshot,
  title={One-Shot RLVR: Reinforcement Learning from Verifiable Rewards with Minimal Data},
  author={Wang, Ypwang et al.},
  journal={arXiv preprint},
  year={2024}
}
```

```bibtex
@article{yang2024qwen25math,
  title={Qwen2.5-Math Technical Report},
  author={Yang et al.},
  journal={arXiv:2409.12122},
  year={2024}
}
```
