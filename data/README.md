# Data Directory

This public release includes the CLINC150 and Banking77 views from the original
minimal package. StackOverflow text is excluded under the project's
redistribution policy. The schema below also describes the local StackOverflow
layout expected by the training adapters.

Dataset sources:

- CLINC150: https://github.com/clinc/oos-eval
- Banking77: https://github.com/PolyAI-LDN/task-specific-datasets
- StackOverflow (obtain separately): https://github.com/jacoxu/StackOverflow

Only seed 42 views are included. These are the existing packaged views; the
release does not establish equivalence to every dataset view used in the
current manuscript.

## Structure

```
data/
├── clinc150/           # CLINC150 dataset (150 intents, multi-domain)
├── stackoverflow/      # Local data only; excluded from Git
└── banking77/          # Banking77 dataset (77 intents, banking)
```

Each dataset contains three KIR (Known Intent Ratio) splits:

```
{dataset}/
├── kir25_seed42/       # KIR=0.25 (25% intents as known)
├── kir50_seed42/       # KIR=0.50 (50% intents as known)
└── kir75_seed42/       # KIR=0.75 (75% intents as known)
```

Each KIR split has the same internal structure:

```
kir{XX}_seed42/
├── gate/               # Gate stage data (OOS detection)
│   ├── train.json
│   ├── val.json
│   └── test.json
├── router/             # Router stage data (domain classification)
│   ├── train.json
│   ├── val.json
│   └── test.json
├── experts/            # Expert stage data (intent classification per domain)
│   ├── {domain_1}/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── {domain_2}/
│   └── ...
└── KNOWN_INTENTS.json  # List of known intents for this split
```

## Data Format

Each JSON file is a list of records:

```json
{
  "text": "change my amex pin to 1234",
  "intent": "pin_change",
  "domain": "banking",
  "split": "train",
  "label": 0
}
```

- `label=0` for known-intent samples, `label=1` for OOS samples.
- `domain` corresponds to the expert module that handles this intent.

## KIR (Known Intent Ratio)

KIR controls the proportion of intents treated as known during training:

```
KIR = |known intents| / |all intents|
```

- KIR=0.25: 25% of intents are known, 75% are held out as OOS
- KIR=0.50: 50% of intents are known, 50% are held out as OOS
- KIR=0.75: 75% of intents are known, 25% are held out as OOS
