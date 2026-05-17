# mSigLIP-CLoRA

Official implementation for the paper:

**A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search**

The repository contains the training and evaluation code for the paper method:
mSigLIP with LoRA, auxiliary Cross-modal Circle Loss, and a curriculum
hard-mining schedule. It is a paper-focused copy of the training code and keeps
only the components needed to reproduce the reported Circle Loss experiments.

## Method

mSigLIP-CLoRA builds on TBPS-mSigLIP with:

- `N-ITC`: sigmoid image-text contrastive alignment with MVS augmentation.
- `Cross-modal Circle Loss`: auxiliary hard-negative mining on normalized
  image/text embeddings.
- `C-ITC`: cyclic image-text consistency regularization.
- `SimCLR`: self-supervised visual consistency.
- `LoRA`: parameter-efficient adaptation on attention projections.

The training objective is:

```text
L = 1.0 * N-ITC + alpha_5(t) * Circle + 0.1 * C-ITC + 0.4 * SimCLR
```

Circle Loss uses a curriculum schedule:

| Epoch | Circle weight |
|---|---:|
| 0-5 | 0.0 |
| 6-20 | linear ramp to 0.1 |
| 21-60 | 0.1 |

## Main Result

| Dataset | Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---:|---:|---:|---:|---:|
| 3000VnPersonSearch | LoRA + Curriculum Circle | 52.28 | 79.55 | 88.03 | 57.32 | 50.57 |
| PRW-TPS-CN | mSigLIP-CLoRA | 59.35 | 70.58 | 75.48 | 46.44 | 15.10 |
| 10% CUHK-PEDES | mSigLIP-CLoRA | 57.10 | 76.98 | 84.34 | 50.90 | 34.85 |

## Repository Layout

```text
trainer.py                 # Hydra training entry point
test.py                    # Evaluation entry point
lightning_data.py          # TBPSDataModule
lightning_models.py        # Lightning module
model/
  tbps.py                  # Forward pass and loss routing
  objectives.py            # N-ITC, Circle, C-ITC, SimCLR
  lora.py                  # PEFT LoRA wrapper
  siglip/                  # Local mSigLIP implementation
config/
  cir_msiglip.yaml         # Main experiment config
  loss/cir_msiglip.yaml    # Paper loss settings
  lora/default.yaml        # LoRA r=32, alpha=64
data/                      # Dataset loaders
solver/                    # Optimizer and scheduler
utils/                     # Metrics and utilities
docs/                      # Architecture and experiment summary
paper/                     # LaTeX source and paper figures
```

## Setup

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

The default backbone config expects multilingual SigLIP weights at:

```text
m_siglip_checkpoints/model.safetensors
```

Dataset paths are configured through Hydra in `config/dataset/*.yaml`.

## Training

Run the reported LoRA + curriculum Circle Loss configuration:

```bash
./run_cir_loss.sh
```

Equivalent Hydra entry point:

```bash
uv run trainer.py -cn cir_msiglip \
  trainer.max_epochs=60 \
  trainer.accumulate_grad_batches=3 \
  ++trainer.precision=16-mixed \
  optimizer=cir_test \
  optimizer.param_groups.default.lr=1e-4 \
  +lora=default
```

For full fine-tuning without LoRA:

```bash
./run_full_finetune.sh
```

## Paper

The paper source is under `paper/`. From that directory:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Notes

Large artifacts are intentionally not versioned: datasets, checkpoints, model
weights, logs, generated outputs, and virtual environments.
