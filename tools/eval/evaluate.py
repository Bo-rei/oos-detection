"""Evaluate a trained Gate, or the complete Gate -> Router -> Expert cascade."""
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
from src.gate.representation import build_racal_model, encode_rows
from src.gate.boundary import _replay_locked
from src.evaluation.metrics import compute_binary_oos_metrics, compute_metrics
from src.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True, help='Local MiniLM directory')
    parser.add_argument('--gate-checkpoint', type=Path, required=True)
    parser.add_argument('--smollm-path', type=Path)
    parser.add_argument('--router-checkpoint', type=Path)
    parser.add_argument('--experts-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/metrics.json'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    supplied = [args.smollm_path, args.router_checkpoint, args.experts_root]
    if any(x is not None for x in supplied) and not all(x is not None for x in supplied):
        parser.error('Full cascade evaluation requires all three downstream path arguments.')
    settings = json.loads(args.config.read_text())
    rows = {split: json.loads((args.data_root / 'gate' / f'{split}.json').read_text())
            for split in ('train', 'test')}
    if any(int(row['label']) != 0 for row in rows['train']):
        raise ValueError('Gate fitting requires Known-only training rows.')
    checkpoint = torch.load(args.gate_checkpoint, map_location='cpu', weights_only=True)
    hidden = int(checkpoint['model']['projection.fc1.weight'].shape[0])
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = build_racal_model(args.model_path, checkpoint['mode'], hidden).to(device)
    model.load_state_dict(checkpoint['model'])
    values = {split: encode_rows(model, tokenizer, content, device, args.batch_size, 256)
              for split, content in rows.items()}
    _, output = _replay_locked(values['train'], rows['train'], values['test'], settings['geometry'])
    scores = output['score']
    threshold = float(settings['geometry']['threshold'])
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if args.smollm_path is None:
        result = compute_binary_oos_metrics([int(row['label']) for row in rows['test']], scores, threshold)
        result['scope'] = 'Gate-only'
    else:
        paths = PipelinePaths(
            model_path=args.smollm_path, gate_encoder_path=args.model_path,
            gate_detector_path=args.output.parent / 'unused_detector.json',
            router_ckpt_path=args.router_checkpoint, experts_root=args.experts_root,
            experts_data_root=args.data_root / 'experts',
            router_data_path=args.data_root / 'router/train.json',
            gate_train_path=args.data_root / 'gate/train.json',
        )
        pipeline = HiLSAMoEV19Pipeline(paths, device=args.device, max_length=64,
                                     semantic_gate_enabled=False)
        pipeline._load_tokenizer()
        pipeline._load_router()
        pipeline._load_domain_mapping()
        pipeline._index_experts()
        predictions = [dict(is_oos=True, gate_pred=1, gate_score=float(score)) for score in scores]
        accepted = np.flatnonzero(scores <= threshold)
        if len(accepted):
            texts = [rows['test'][i]['text'] for i in accepted]
            routed = pipeline._router_predict(texts, batch_size=args.batch_size)
            domains = [pipeline.domain_id_to_name[i] for i in routed['domain_ids']]
            for domain in sorted(set(domains)):
                ids = [i for i, item in enumerate(domains) if item == domain]
                expert = pipeline._expert_predict_group(domain, [texts[i] for i in ids], batch_size=args.batch_size)
                for j, local in enumerate(ids):
                    index = int(accepted[local])
                    predictions[index].update(is_oos=False, gate_pred=0, domain=domain,
                                              intent=expert['intent_names'][j])
        result, stages = compute_metrics(rows['test'], predictions)
        result.pop('known_macro_f1', None)
        known = sorted({row['intent'] for row in rows['test'] if int(row['label']) == 0})
        truth = ['__oos__' if int(row['label']) else row['intent'] for row in rows['test']]
        predicted = ['__oos__' if row['is_oos'] else row['intent'] for row in predictions]
        result['known_f1'] = float(f1_score(truth, predicted, labels=known, average='macro', zero_division=0))
        result.update(scope='Full cascade', stage_counts=stages)
    result.update(dataset=settings['dataset'], kir=settings['kir'], seed=settings['seed'],
                  threshold=threshold, test_used_for_selection=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
