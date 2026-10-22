# Multi-cluster Boundary Learning for OOS Intent Detection

Implementation snapshot for *A Multi-cluster Boundary Learning Method for Out-of-Scope Intent Detection via MiniLM Embedding*.

## Release scope

This initial release preserves the previously prepared minimal code package,
including the Gate, Router, Expert, and baseline implementations. The bundled
Gate entrypoint fits geometric boundaries on pretrained MiniLM embeddings; it
does not include the learnable MiniLM representation training described in the
current manuscript. The bundled data covers seed 42 only. Full training and
evaluation of this standalone release have not been verified, so this snapshot
does not establish reproduction of the current paper's three-seed results.

## Download

```bash
git clone https://github.com/Bo-rei/oos-detection.git
cd oos-detection
```

## Requirements

```
pip install -r requirements.txt
```

## Data

The `data/` directory includes the prepared CLINC150 and Banking77 views from
the original package, with KIR splits (kir25/50/75) for seed 42. StackOverflow
text is excluded from this public release under the project's redistribution
policy; obtain it separately and prepare the same local layout. See
`data/README.md` for the format and dataset sources. Pretrained model weights
and generated training outputs are also excluded.

## Training

```bash
# 1. Gate (geometric, no neural network)
python tools/gate/train_multisphere_corrected.py \
    --data_root data/clinc150/kir50_seed42 \
    --model_path <MiniLM_PATH>

# 2. Router (SmolLM-135M + LoRA)
python tools/train/train_router_v19.py \
    --model_path <SmolLM_PATH> \
    --data_dir data/clinc150/kir50_seed42/router

# 3. Expert (SmolLM-135M + LoRA, per domain)
python tools/train/train_expert_v19.py \
    --domain banking \
    --model_path <SmolLM_PATH> \
    --data_dir data/clinc150/kir50_seed42/experts
```

## Baselines

The bundled baseline adapter trains MSP, OpenMax, DOC, DeepUnk, KNNCL, ADB,
and DA-ADB on the same data layout:

```bash
python baselines/train_baselines.py \
    --method ADB \
    --dataset clinc150 \
    --kir 0.50 \
    --seed 42 \
    --data-root data \
    --model-root <BERT_PATH> \
    --output-root outputs/adb
```

MOGB is included as another baseline with its training adapter:

```bash
python baselines/train_mogb.py --dataset clinc150 --kir 0.50 --seed 42 \
    --data-root data
```

## Inference

```python
from pathlib import Path
from src.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths

paths = PipelinePaths(
    model_path=Path("<SmolLM_PATH>"),
    gate_encoder_path=Path("<MiniLM_PATH>"),
    gate_detector_path=Path("outputs/gate/detector.json"),
    router_ckpt_path=Path("outputs/router/best_model.pt"),
    experts_root=Path("outputs/experts"),
    experts_data_root=Path("data/clinc150/kir50_seed42/experts"),
    router_data_path=Path("data/clinc150/kir50_seed42/router/train.json"),
    gate_train_path=Path("data/clinc150/kir50_seed42/gate/train.json"),
)
pipeline = HiLSAMoEV19Pipeline(paths, device="cpu")
pipeline.load()
results = pipeline.predict_batch(["what is my account balance"])
```

## Baseline sources

The baseline runtime under `baselines/open_intent_detection/` is adapted from
[TEXTOIR](https://github.com/thuiar/TEXTOIR); its MIT license is retained in that
directory. The MOGB implementation and adapter are based on
[MOGB](https://github.com/Liyanhuaa/MOGB). Original source notices are retained
in the bundled files. Baseline dependencies and native OpenMax/libMR components
may require their upstream installation steps.
