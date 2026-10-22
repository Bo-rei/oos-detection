# Local data inputs

Dataset text is not included in Git. Obtain data from the original sources:

- CLINC150: https://github.com/clinc/oos-eval
- Banking77: https://github.com/PolyAI-LDN/task-specific-datasets
- StackOverflow: https://github.com/jacoxu/StackOverflow

The current configurations use historical v19 prepared views for CLINC150
and StackOverflow, and the standard TextOIR-aligned 77-intent view for
Banking77. Official downloads alone do not establish those exact sample
splits. Use the matching prepared views to replay the supplied settings;
`banking77_oos` is not a substitute for standard `banking77`.

Each local `data/<dataset>/kirXX_seedYY/` directory contains:

```text
gate/{train,val,test}.json
router/{train,val}.json
router/domain_map.json
experts/<domain>/{train,val}.json
experts/<domain>/intent_map.json
KNOWN_INTENTS.json
```

Gate JSON files are lists of records with `text`, `intent`, `domain`, and
`label` (0 for Known, 1 for OOS); `sample_id` is optional. Router and Expert
labels are the corresponding domain/intent class indices. Training and
validation for the released trainers use Known samples only. Evaluation
reads the complete Gate test split.

If you already have the matching prepared Gate JSON files, create downstream
views without changing their sample split or selecting a new Known list:

```bash
python tools/data/prepare_data.py \
  --source /path/to/prepared/clinc150/kir50_seed42 \
  --config configs/clinc150/kir50_seed42.json \
  --output data/clinc150/kir50_seed42
```

The adapter preserves source row order and uses the Known-intent list in
the selected configuration. It does not download or resplit raw datasets.
