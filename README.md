# Multi-cluster Boundary Learning for OOS Intent Detection

Implementation of *A Multi-cluster Boundary Learning Method for Out-of-Scope
Intent Detection via MiniLM Embedding*.

The Gate adapts MiniLM on Known intents using a residual projection and
prototype classification, compactness, and separation losses. A geometric
boundary rejects OOS inputs; accepted inputs pass through a SmolLM Router and
domain Experts.

## Install

Use Python 3.10+ and install `requirements.txt`. Obtain local copies of
`sentence-transformers/all-MiniLM-L6-v2` and `HuggingFaceTB/SmolLM-135M`.
Baselines additionally require BERT; OpenMax/libMR has native dependencies
and should follow the bundled upstream installation instructions.

## Layout

- `src/gate/`: trainable MiniLM, losses, geometric boundaries, and scoring.
- `src/router/`, `src/models/`, `src/pipeline/`: downstream classification.
- `tools/`: data adaptation, training, and evaluation entrypoints.
- `configs/`: current per-dataset settings for KIR 0.25/0.50/0.75 and seeds
  13/42/87, including Known-intent lists and locked boundary settings.
- `baselines/`: MSP, OpenMax, DOC, DeepUnk, KNNCL, ADB, DA-ADB, and MOGB.

The repository contains code, configuration, and documentation. Data text,
model weights, experiment outputs, and the manuscript are supplied separately.
See `data/README.md` for the input contract and dataset sources.

## Train

Commands below use one prepared dataset/KIR/seed directory. Select its
matching configuration; each dataset retains its current training schedule.
CLINC150/StackOverflow use the historical prepared views; Banking77 uses the
standard 77-intent TextOIR-aligned view. Historical Banking77-OOS is a separate
dataset and must not be substituted.

```bash
python tools/gate/train_gate.py \
  --config configs/clinc150/kir50_seed42.json \
  --data-root data/clinc150/kir50_seed42 \
  --model-path models/all-MiniLM-L6-v2 \
  --output-dir outputs/gate --device cuda

python tools/train/train_router.py \
  --model_path models/smollm135m \
  --data_dir data/clinc150/kir50_seed42/router \
  --output_dir outputs/router --defer_test

# Repeat for each domain in experts/.
python tools/train/train_expert.py --domain banking \
  --model_path models/smollm135m \
  --data_dir data/clinc150/kir50_seed42/experts \
  --output_dir outputs/experts --defer_test
```

Gate training reads only Known training/validation rows. Single-domain
datasets use constant routing during evaluation.

## Evaluate

```bash
python tools/eval/evaluate.py \
  --config configs/clinc150/kir50_seed42.json \
  --data-root data/clinc150/kir50_seed42 \
  --model-path models/all-MiniLM-L6-v2 \
  --gate-checkpoint outputs/gate/checkpoint.pt \
  --smollm-path models/smollm135m \
  --router-checkpoint outputs/router/best_model.pt \
  --experts-root outputs/experts \
  --output outputs/metrics.json --device cuda
```

Omit the three downstream path arguments to evaluate the Gate alone. The
output labels Gate-only and full-cascade metrics explicitly; Known-class
macro F1 in the cascade is computed over the full test set.

## Baselines

```bash
python baselines/train_baselines.py --method ADB --dataset clinc150 \
  --kir 0.50 --seed 42 --data-root data \
  --model-root models/bert-base-uncased --output-root outputs/adb

python baselines/train_mogb.py --dataset clinc150 --kir 0.50 --seed 42 \
  --data-root data --model-root models/bert-base-uncased \
  --output-root outputs/mogb
```

Baseline implementations are adapted from
[TEXTOIR](https://github.com/thuiar/TEXTOIR) and
[MOGB](https://github.com/Liyanhuaa/MOGB). Upstream notices are retained; the
TEXTOIR MIT license is included in `baselines/open_intent_detection/LICENSE`.

Validation in an isolated copy covered seven CLI entrypoints, exact loss and
gradient agreement, boundary-score agreement across 27 configurations, and
checkpoint-based Gate and full-cascade inference on a small local sample.

This release exports the current project implementations and locked
configurations. Full training has not been rerun for this packaged release; exact
reproduction of every paper result is not established by the packaging checks.
