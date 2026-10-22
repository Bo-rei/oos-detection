"""Train the current MiniLM representation using Known training/validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gate.representation import build_racal_model, encode_rows, set_seed
from src.gate.training import (
    TrainConfig, _center_losses, _class_centers, configure, run_trainable,
)


def train_recipe(settings, views, model_path, output, device):
    """The current recipe loop, with Known-validation macro F1 selection."""
    recipe = settings['recipe']
    seed = int(settings['seed'])
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    mode = 'last2_minilm_plus_projection'
    model = build_racal_model(model_path, mode, recipe['projection']).to(device)
    train, validation = views['train'], views['val']
    names = sorted({row['intent'] for row in train})
    mapping = {name: index for index, name in enumerate(names)}
    target = torch.tensor([mapping[row['intent']] for row in train], device=device)
    gold = np.array([mapping[row['intent']] for row in validation])
    best, history = -1., []
    for epoch in range(1, recipe['epochs'] + 1):
        if epoch in (1, 2):
            optimizer = configure(model, recipe, epoch == 1)
        values = encode_rows(model, tokenizer, train, device, 128, 256)
        centers = torch.as_tensor(_class_centers(values, train)[0], device=device)
        model.train()
        losses = []
        order = np.random.default_rng(seed + epoch * 7919).permutation(len(train))
        for start in range(0, len(order), 64):
            indices = order[start:start+64]
            tokens = tokenizer([train[int(i)]['text'] for i in indices], padding=True,
                               truncation=True, max_length=256, return_tensors='pt').to(device)
            loss, _ = _center_losses(
                model(tokens), target[torch.as_tensor(indices, device=device)], centers,
                recipe['temperature'], recipe['intra'], recipe['inter'], 1., recipe['margin'],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        values = encode_rows(model, tokenizer, train, device, 128, 256)
        centers_np = _class_centers(values, train)[0]
        val = encode_rows(model, tokenizer, validation, device, 128, 256)
        predicted = np.argmax(val @ centers_np.T, axis=1)
        score = float(f1_score(gold, predicted, average='macro'))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), known_validation_f1=score))
        if score > best + 1e-12:
            best, best_epoch = score, epoch
            torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            mode=mode, epoch=epoch, recipe=recipe), output / 'checkpoint.pt')
        (output / 'training_history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f'epoch={epoch} Known-F1={score:.6f}', flush=True)
    return dict(epoch=best_epoch, known_validation_f1=best, test_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='One prepared dataset/KIR/seed directory')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    known = set(settings['known_intents'])
    views = {}
    for split in ('train', 'val'):
        rows = json.loads((args.data_root / 'gate' / f'{split}.json').read_text())
        if split == 'train' and any(row['intent'] not in known or int(row['label']) != 0 for row in rows):
            raise ValueError('Representation training requires Known-only rows.')
        rows = [dict(row, sample_id=str(row.get('sample_id', f'{split}-{i}')))
                for i, row in enumerate(rows) if row['intent'] in known and int(row['label']) == 0]
        if not rows:
            raise ValueError(f'Empty Known {split} split.')
        views[split] = rows
    if {row['intent'] for row in views['train']} != known:
        raise ValueError('Training intent list differs from the selected configuration.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if settings['schedule'] == 'historical':
        recipe = settings['recipe']
        config = TrainConfig(
            model_path=str(args.model_path), dataset=settings['dataset'],
            data_root=str(args.data_root), seed=int(settings['seed']),
            warmup_epochs=recipe['warmup_epochs'], finetune_epochs=recipe['finetune_epochs'],
            patience=recipe['patience'], device=args.device,
            temperature=recipe['temperature'], intra_weight=recipe['intra'], inter_weight=recipe['inter'],
        )
        result = run_trainable(config, {}, views, args.output_dir)
    else:
        result = train_recipe(settings, views, args.model_path, args.output_dir, device)
    (args.output_dir / 'config.json').write_text(json.dumps(settings, indent=2) + '\n')
    (args.output_dir / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
