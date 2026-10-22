"""Adapt a local prepared Gate JSON split for the released cascade trainers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Directory containing gate/{train,val,test}.json')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    known = set(config['known_intents'])
    views = {}
    for split in ('train', 'val', 'test'):
        rows = json.loads((args.source / 'gate' / f'{split}.json').read_text())
        views[split] = [dict(row, label=int(str(row['intent']) not in known),
                             sample_id=str(row.get('sample_id', f'{split}-{i}')))
                        for i, row in enumerate(rows)]
        if split != 'test':
            views[split] = [row for row in views[split] if row['label'] == 0]
        if not views[split]:
            raise ValueError(f'Empty {split} split.')
    if {row['intent'] for row in views['train']} != known:
        raise ValueError('Source intent list differs from the selected configuration.')
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in views.items():
        write(args.output / 'gate' / f'{split}.json', rows)
    write(args.output / 'KNOWN_INTENTS.json', {'known_intents': sorted(known)})
    domains = sorted({row['domain'] for row in views['train']})
    domain_map = {domain: i for i, domain in enumerate(domains)}
    write(args.output / 'router/domain_map.json', domain_map)
    for split in ('train', 'val'):
        write(args.output / 'router' / f'{split}.json',
              [dict(row, label=domain_map[row['domain']]) for row in views[split]])
    for domain in domains:
        intents = sorted({row['intent'] for row in views['train'] if row['domain'] == domain})
        mapping = {intent: i for i, intent in enumerate(intents)}
        write(args.output / 'experts' / domain / 'intent_map.json', mapping)
        for split in ('train', 'val'):
            write(args.output / 'experts' / domain / f'{split}.json',
                  [dict(row, label=mapping[row['intent']]) for row in views[split] if row['domain'] == domain])
    print('Prepared', args.output)


if __name__ == '__main__':
    main()
