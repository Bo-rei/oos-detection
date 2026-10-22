"""Pinned MOGB methods with shared data and a deferred, common evaluator.

No training-loop, loss, clustering, radius or classification replacement.
"""
import argparse
import contextlib
import gc
import importlib.util
import io
import json
import numpy as np
import os
from pathlib import Path
import subprocess
import sys
import traceback
import tempfile
import signal
import time

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'open_intent_detection'), str(ROOT / 'open_intent_detection'), str(ROOT / 'mogb_support'), str(ROOT / 'mogb_impl')]

ART = ROOT.parent / 'artifacts' / 's2c'


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n')

OUT = ART.parent / 'mogb_baseline'
OFFICIAL = ROOT / 'mogb_impl'
LEGACY = ROOT / 'mogb_support'
DIRECT_BLOCK = 4096


def _aligned_view(size):
    storage = np.empty(size + DIRECT_BLOCK, dtype=np.uint8)
    offset = (-int(storage.ctypes.data)) % DIRECT_BLOCK
    view = storage[offset:offset + size]
    assert int(view.ctypes.data) % DIRECT_BLOCK == 0
    return storage, view


def _direct_write(path, payload):
    if not hasattr(os, 'O_DIRECT'):
        raise RuntimeError('Linux O_DIRECT is unavailable; refusing buffered offload')
    padded = ((len(payload) + DIRECT_BLOCK - 1) // DIRECT_BLOCK) * DIRECT_BLOCK
    storage, view = _aligned_view(padded)
    view.fill(0)
    view[:len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_DIRECT, 0o600)
    try:
        written = 0
        while written < padded:
            count = os.write(fd, memoryview(view)[written:])
            if count <= 0:
                raise OSError('short O_DIRECT write')
            written += count
    finally:
        os.close(fd)
    del view, storage
    return len(payload), padded


def _direct_read(path, payload_size, padded_size):
    storage, view = _aligned_view(padded_size)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        read = 0
        while read < padded_size:
            count = os.readv(fd, [memoryview(view)[read:]])
            if count <= 0:
                raise OSError('short O_DIRECT read')
            read += count
    finally:
        os.close(fd)
    payload = view[:payload_size].tobytes()
    del view, storage
    return payload


class DirectTensorStore:
    """One aligned O_DIRECT file reused for one epoch at a time."""

    def __init__(self, directory):
        if not hasattr(os, 'O_DIRECT'):
            raise RuntimeError('Linux O_DIRECT is unavailable; refusing buffered offload')
        fd, name = tempfile.mkstemp(dir=directory, suffix='.bin')
        os.close(fd)
        self.path = Path(name)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC | os.O_DIRECT, 0o600)
        self.offset = 0

    def write(self, payload):
        padded = ((len(payload) + DIRECT_BLOCK - 1) // DIRECT_BLOCK) * DIRECT_BLOCK
        storage, view = _aligned_view(padded)
        view.fill(0)
        view[:len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        offset = self.offset
        written = 0
        while written < padded:
            count = os.pwrite(self.fd, memoryview(view)[written:], offset + written)
            if count <= 0:
                raise OSError('short O_DIRECT store write')
            written += count
        self.offset += padded
        del view, storage
        return offset, len(payload), padded

    def read(self, offset, payload_size, padded_size):
        storage, view = _aligned_view(padded_size)
        read = 0
        while read < padded_size:
            count = os.preadv(self.fd, [memoryview(view)[read:]], offset + read)
            if count <= 0:
                raise OSError('short O_DIRECT store read')
            read += count
        payload = view[:payload_size].tobytes()
        del view, storage
        return payload

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)

    def reset_epoch(self):
        """Drop the completed epoch without retaining its disk footprint."""
        if self.fd is None:
            raise RuntimeError('cannot reset a closed direct tensor store')
        os.ftruncate(self.fd, 0)
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class DirectSavedTensor:
    """Autograd saved-tensor hook using O_DIRECT, without page-cache growth."""

    def __init__(self, tensor, store):
        self.device = tensor.device
        stream = io.BytesIO()
        np.save(stream, tensor.detach().cpu().numpy(), allow_pickle=False)
        self.store = store
        self.offset, self.payload_size, self.padded_size = store.write(stream.getvalue())

    def unpack(self):
        payload = self.store.read(self.offset, self.payload_size, self.padded_size)
        value = np.load(io.BytesIO(payload), allow_pickle=False)
        return torch.from_numpy(value).to(self.device)


class CpuSavedTensor:
    """Autograd saved-tensor hook matching torch.save_on_cpu semantics."""

    def __init__(self, tensor):
        self.device = tensor.device
        # A saved-tensor hook only needs the value for backward.  Detaching
        # the CPU snapshot prevents the compatibility object from retaining
        # the original GPU-side autograd graph through CopyBackwards.
        self.value = tensor.detach().cpu()

    def unpack(self):
        return self.value.to(self.device)


class CpuFp16SavedTensor:
    """CPU saved-tensor hook with bounded storage for large float activations.

    This changes only saved-activation precision during backward; the model,
    official losses, clustering and optimizer remain unchanged.  It is kept
    separate from CpuSavedTensor so manifests can distinguish exact CPU
    offload from this official-compatible memory adaptation.
    """

    def __init__(self, tensor):
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.compressed = bool(
            tensor.is_floating_point()
            and tensor.dtype == torch.float32
            and tensor.numel() >= 1024
        )
        value = tensor.detach().cpu()
        self.value = value.to(torch.float16) if self.compressed else value

    def unpack(self):
        value = self.value.to(self.device)
        return value.to(self.dtype) if self.compressed else value


class CpuInt8SavedTensor:
    """CPU saved-tensor hook using per-tensor symmetric int8 quantization.

    This is a throughput-oriented compatibility fallback for hosts that
    cannot hold the complete official feature graph in RAM.  It leaves the
    official forward, clustering, losses and optimizer intact, but introduces
    backward-value quantization and must therefore never be called strict.
    """

    def __init__(self, tensor):
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.compressed = bool(
            tensor.is_floating_point()
            and tensor.dtype == torch.float32
            and tensor.numel() >= 1024
        )
        value = tensor.detach().cpu()
        if self.compressed:
            maximum = float(value.abs().amax().item())
            self.scale = max(maximum / 127.0, 1e-12)
            self.value = torch.clamp(
                torch.round(value / self.scale), -127, 127
            ).to(torch.int8)
            del value
        else:
            self.scale = None
            self.value = value

    def unpack(self):
        if not self.compressed:
            return self.value.to(self.device)
        return (self.value.to(self.device, dtype=torch.float32) * self.scale).to(
            self.dtype
        )


def _unpack_saved_tensor(value):
    if isinstance(value, (DirectSavedTensor, CpuSavedTensor, CpuFp16SavedTensor,
                          CpuInt8SavedTensor)):
        return value.unpack()
    return value


@contextlib.contextmanager
def _feature_only_saved_tensors(model, feature_pack):
    """Offload only graphs retained by the official feature-memory bank."""
    phase = {'feature': False}
    model_module = model.module if hasattr(model, 'module') else model
    model_class = type(model_module)
    original_forward = model_class.forward

    def phase_forward(self, *args, **kwargs):
        feature_ext = kwargs.get('feature_ext', False)
        phase['feature'] = feature_ext not in (False, None, 'False')
        return original_forward(self, *args, **kwargs)

    def pack(value):
        if phase['feature']:
            return feature_pack(value)
        return value

    model_class.forward = phase_forward
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, _unpack_saved_tensor):
            yield
    finally:
        model_class.forward = original_forward


@contextlib.contextmanager
def _last_layer_checkpointing(model):
    """Checkpoint only the trainable BERT layer, leaving frozen layers exact."""
    model_module = model.module if hasattr(model, 'module') else model
    target_layer = model_module.bert.encoder.layer[-1]
    layer_class = type(target_layer)
    original_forward = layer_class.forward

    def checkpointed_forward(self, *args, **kwargs):
        if self is not target_layer:
            return original_forward(self, *args, **kwargs)

        def recompute(*values):
            return original_forward(self, *values, **kwargs)

        return checkpoint(recompute, *args, use_reentrant=False)

    layer_class.forward = checkpointed_forward
    try:
        yield
    finally:
        layer_class.forward = original_forward


def _memory_safe_official_train(manager, args, data, official, epoch_cleanup=None):
    """Upstream train loop with explicit lifetime cleanup only.

    The forward, loss, optimizer, clustering and dev-selection operations are
    copied from the pinned upstream loop.  The extra deletes prevent the
    epoch-end feature/cluster graph from overlapping the next epoch, and the
    old best-model object is collected after replacement.
    """
    wait = 0
    best_model = None
    for epoch in official.trange(int(args.num_train_epochs), desc='Epoch'):
        manager.model.train()
        trace_memory(f'epoch_{epoch + 1}_start')
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        memory_bank = []
        memory_bank_label = []

        for step, batch in enumerate(official.tqdm(data.train_dataloader, desc='Iteration')):
            batch = tuple(t.to(manager.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            batch_number = len(data.train_dataloader)
            with torch.set_grad_enabled(True):
                loss1 = manager.model(input_ids, segment_ids, input_mask, label_ids,
                                      mode='train')

                manager.optimizer.zero_grad()
                loss1.backward()
                manager.optimizer.step()
                tr_loss += loss1.item()
                official.util.summary_writer.add_scalar(
                    'Loss/loss1', loss1.item(), step + epoch * batch_number)
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                features = manager.model(input_ids, segment_ids, input_mask,
                                         feature_ext='True')
                memory_bank.append(features.cpu())
                memory_bank_label.append(label_ids.cpu())

            if (step + 1) == batch_number:
                accumulated_features = torch.cat(memory_bank, dim=0).to('cuda:0')
                accumulated_labels = torch.cat(memory_bank_label, dim=0).to('cuda:0')
                manager.gb_centroids, manager.gb_radii, manager.gb_labels, loss2 = (
                    manager.clusterLoss.forward(args, accumulated_features,
                                                accumulated_labels, select=False))
                manager.optimizer2.zero_grad()
                loss2.backward()
                official.util.summary_writer.add_scalar(
                    'Loss/loss11', loss2.item(), step + epoch * batch_number)
                manager.optimizer2.step()

                # These names are live in the upstream function until the
                # next epoch.  Release them before dev evaluation.
                del loss1, features, loss2, accumulated_features, accumulated_labels
                memory_bank = []
                memory_bank_label = []

        loss = tr_loss / nb_tr_steps
        print('train_loss', loss)
        if epoch_cleanup is not None:
            # All feature/cluster references have been deleted above.  Reset
            # the backing store before validation so its size is bounded by a
            # single epoch rather than by the whole training run.
            gc.collect()
            epoch_cleanup()
        eval_score = manager.eval(args, data)
        print('eval_score', eval_score)
        if eval_score > manager.best_eval_score:
            old_best = best_model
            best_model = official.copy.deepcopy(manager.model)
            del old_best
            gc.collect()
            torch.cuda.empty_cache()
            wait = 0
            manager.best_eval_score = eval_score
        else:
            wait += 1
            if wait >= args.wait_patient:
                break

    manager.model = best_model
    if args.save_model:
        manager.save_model(args)
    return manager.gb_centroids, manager.gb_radii, manager.gb_labels


def _recompute_feature_graph_official_train(manager, args, data, official):
    """Run the official objective without retaining one graph per sample.

    The released loop first performs one CE update per minibatch, stores every
    feature graph, and applies one epoch-level cluster-loss update.  The
    feature tensors and the cluster centres are independent of the autograd
    graph, so this compatibility path records the post-CE trainable parameter
    state and RNG state for each feature extraction, computes the official
    balls from detached features, then recomputes each graph one at a time.
    Per-minibatch cluster gradients are weighted by minibatch size, exactly
    reproducing the mean used by ``clusterLoss.compute_classification_loss``;
    ``optimizer2.step`` remains a single epoch-level update.
    """
    wait = 0
    best_model = None
    model_module = manager.model.module if hasattr(manager.model, 'module') else manager.model
    trainable = [(name, parameter) for name, parameter in model_module.named_parameters()
                 if parameter.requires_grad]

    def trace_memory(label):
        if os.environ.get('MOGB_MEMORY_TRACE') != '1':
            return
        torch.cuda.synchronize(manager.device)
        live = 0
        live_bytes = 0
        largest = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    live += 1
                    size = obj.numel() * obj.element_size()
                    live_bytes += size
                    largest.append((size, tuple(obj.shape), str(obj.dtype),
                                    bool(obj.requires_grad), obj))
            except Exception:
                pass
        largest = sorted(largest, key=lambda item: item[0], reverse=True)[:8]
        largest_info = [item[:4] for item in largest]
        ref_info = []
        if os.environ.get('MOGB_MEMORY_TRACE_REFS') == '1':
            for _, shape, _, _, tensor in largest[:3]:
                refs = gc.get_referrers(tensor)
                details = []
                for ref in refs[:12]:
                    if isinstance(ref, tuple):
                        owners = []
                        for owner in gc.get_referrers(ref)[:8]:
                            if isinstance(owner, dict):
                                owners.append('dict:' + ','.join(
                                    str(key) for key, value in owner.items()
                                    if value is ref)[:120])
                            else:
                                owners.append(type(owner).__name__)
                        details.append(('tuple', len(ref), owners))
                    else:
                        details.append(type(ref).__name__)
                ref_info.append((shape, details))
        print('memory_trace', label,
              'allocated', int(torch.cuda.memory_allocated()),
              'reserved', int(torch.cuda.memory_reserved()),
              'live_tensors', live, 'live_bytes', live_bytes,
              'largest', largest_info, 'referrers', ref_info, flush=True)

    def cpu_model_state():
        # ``copy.deepcopy(model)`` keeps every best checkpoint on CUDA until
        # the end of training.  A CPU state_dict is value-equivalent for the
        # official checkpoint contract and prevents one model-sized GPU leak
        # per improving epoch.
        return {name: value.detach().cpu().clone()
                for name, value in model_module.state_dict().items()}

    def snapshot_parameters():
        return tuple(parameter.detach().cpu().clone() for _, parameter in trainable)

    def restore_parameters(snapshot):
        for (_, parameter), value in zip(trainable, snapshot):
            parameter.data.copy_(value.to(parameter.device))

    for epoch in official.trange(int(args.num_train_epochs), desc='Epoch'):
        manager.model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        detached_features = []
        detached_labels = []
        records = []

        for step, batch in enumerate(official.tqdm(data.train_dataloader, desc='Iteration')):
            cpu_batch = tuple(value.detach().cpu() for value in batch)
            batch = tuple(value.to(manager.device) for value in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            batch_number = len(data.train_dataloader)
            with torch.set_grad_enabled(True):
                loss1 = manager.model(input_ids, segment_ids, input_mask,
                                      label_ids, mode='train')
                manager.optimizer.zero_grad()
                loss1.backward()
                manager.optimizer.step()
                tr_loss += loss1.item()
                official.util.summary_writer.add_scalar(
                    'Loss/loss1', loss1.item(), step + epoch * batch_number)
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                # The original feature forward is still performed at the
                # original point in the loop, but without an autograd graph.
                # Its value and dropout mask are replayed below.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(manager.device)
                parameter_state = snapshot_parameters()
                with torch.no_grad():
                    features = manager.model(input_ids, segment_ids, input_mask,
                                             feature_ext='True')
                detached_features.append(features.detach().cpu())
                detached_labels.append(label_ids.detach().cpu())
                records.append((cpu_batch, parameter_state, cpu_rng, cuda_rng))
                del loss1, features

        total_examples = sum(int(labels.numel()) for labels in detached_labels)
        trace_memory(f'epoch_{epoch + 1}_after_ce_feature_pass')
        trace_memory('before_cluster')
        accumulated_features = torch.cat(detached_features, dim=0).to(manager.device)
        accumulated_labels = torch.cat(detached_labels, dim=0).to(manager.device)
        manager.gb_centroids, manager.gb_radii, manager.gb_labels, _ = (
            manager.clusterLoss.forward(args, accumulated_features,
                                        accumulated_labels, select=False))
        trace_memory('after_cluster')

        # Keep the parameters produced by the CE pass while replaying the
        # feature graphs.  Only gradients, not graphs, survive each iteration.
        final_parameters = snapshot_parameters()
        manager.optimizer2.zero_grad()
        for cpu_batch, parameter_state, cpu_rng, cuda_rng in records:
            restore_parameters(parameter_state)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device=manager.device)
            input_ids, input_mask, segment_ids, label_ids = (
                value.to(manager.device) for value in cpu_batch)
            with torch.set_grad_enabled(True):
                features = manager.model(input_ids, segment_ids, input_mask,
                                         feature_ext='True')
                loss2 = manager.clusterLoss.compute_classification_loss(
                    features, label_ids, manager.gb_centroids, manager.gb_labels)
                (loss2 * (label_ids.numel() / total_examples)).backward()
            del features, loss2

        restore_parameters(final_parameters)
        official.util.summary_writer.add_scalar(
            'Loss/loss11', 0.0, epoch * len(data.train_dataloader))
        manager.optimizer2.step()
        trace_memory('after_cluster_step')
        del detached_features, detached_labels, records
        del accumulated_features, accumulated_labels
        gc.collect()
        torch.cuda.empty_cache()
        trace_memory('after_epoch_cleanup')

        loss = tr_loss / nb_tr_steps
        print('train_loss', loss)
        eval_score = manager.eval(args, data)
        print('eval_score', eval_score)
        if eval_score > manager.best_eval_score:
            old_best = best_model
            best_model = cpu_model_state()
            del old_best
            gc.collect()
            torch.cuda.empty_cache()
            wait = 0
            manager.best_eval_score = eval_score
        else:
            wait += 1
            if wait >= args.wait_patient:
                break

    if best_model is None:
        raise RuntimeError('official training produced no Known-dev checkpoint')
    model_module.load_state_dict(best_model)
    del best_model
    gc.collect()
    torch.cuda.empty_cache()
    if args.save_model:
        manager.save_model(args)
    return manager.gb_centroids, manager.gb_radii, manager.gb_labels


def run(dataset, kir, seed, offload='disk', gradient_checkpointing=False,
        last_layer_checkpointing=False, memory_safe_loop=False,
        train_batch_size=None, eval_batch_size=None,
        recompute_feature_graphs=False, data_root=None, output_root=None,
        data_dataset=None, model_root=None):
    import transformers
    from final_prediction_metrics import prediction_metrics
    torch.set_num_threads(2)
    tag = f'kir{round(100*kir):02d}_seed{seed}'
    output_base = Path(output_root) if output_root is not None else OUT
    data_base = Path(data_root) if data_root is not None else ART / 'data'
    folder = output_base / dataset / tag
    if (folder / 'result.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    previous = folder / 'MANIFEST.json'
    if previous.exists():
        history_path = folder / 'attempts.json'
        attempts = json.loads(history_path.read_text()) if history_path.exists() else []
        attempts.append(json.loads(previous.read_text()))
        dump(history_path, attempts)
    data_root = data_base / (data_dataset or dataset) / tag
    known_path = data_root / 'known_labels.json'
    if known_path.is_file():
        known = json.loads(known_path.read_text())
    else:
        known = json.loads((data_root / 'KNOWN_INTENTS.json').read_text())['known_intents']
    commit = '5b689e2a03de0d86ec41212825e5db8d7f0e5c02'
    manifest = dict(status='training', method='MOGB-official-compatible',
        dataset=dataset, kir=kir, seed=seed, data_root=str(data_root), known_labels=known,
        upstream='https://github.com/Liyanhuaa/MOGB', commit=commit,
        torch=torch.__version__, transformers=transformers.__version__, python=sys.version,
        device='cuda', real_oos_training=False, real_oos_validation=False,
        test_used_for_selection=False, backbone='bert-base-uncased',
        modifications=['modern BERT API shim', 'original BertAdam 0.6.2 imported directly',
            'shared JSON data loader and label mapping', 'deferred test loader',
            f'autograd saved tensors offloaded to {offload} without changing forward computations',
            'direct mode keeps immediately-backpropagated CE tensors in memory and offloads only persistent feature graphs',
            'direct mode stores aligned payloads in one contiguous per-epoch file',
            'eager attention matching model_pipeline BERT; modern SDPA alignment fails at length 55',
            'save final model and granular balls; common prediction metrics'],
        training_loop='unmodified official PretrainModelManager.train',
        model_selection='official Known dev accuracy and patience', full_pipeline=False,
        task='end_to_end_open_intent_classifier', evaluator='common final prediction metrics',
        train_batch_size=train_batch_size if train_batch_size is not None else 128,
        eval_batch_size=eval_batch_size if eval_batch_size is not None else 64,
        recompute_feature_graphs=recompute_feature_graphs)
    dump(folder / 'MANIFEST.json', manifest)
    def interrupted(signum, frame):
        raise InterruptedError(f'training interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Verify allocation before loading a model; never silently train on CPU.
        torch.zeros(1, device='cuda:0')
        # Use the exact historical optimizer, not the existing approximate AdamW shim.
        sys.path[:0] = [str(ROOT / 'mogb_support'), str(OFFICIAL)]
        import pytorch_pretrained_bert
        spec = importlib.util.spec_from_file_location('pytorch_pretrained_bert.optimization',
                    LEGACY / 'pytorch_pretrained_bert/optimization.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from init_parameter import init_model
        import dataloader
        from pretrain import PretrainModelManager
        from gb_test import ModelManager
        from model import BertForModel
        original_load = BertForModel.from_pretrained
        BertForModel.from_pretrained = lambda *a, **kw: original_load(*a, attn_implementation='eager', **kw)
        runtime = dict(banking77='banking', clinc150='oos', stackoverflow='stackoverflow')[dataset]
        bert_model = Path(model_root) if model_root is not None else ROOT.parent / 'models' / 'bert-base-uncased'
        args = init_model().parse_args(['--dataset', runtime, '--known_cls_ratio', str(kir),
            '--labeled_ratio','1.0','--freeze_bert_parameters','--save_model',
            '--bert_model',str(bert_model),
            '--pretrain_dir',str(folder / 'checkpoint'), '--seed',str(seed)])
        if train_batch_size is not None:
            args.train_batch_size = int(train_batch_size)
        if eval_batch_size is not None:
            args.eval_batch_size = int(eval_batch_size)
        args.max_seq_length = dict(banking77=55, clinc150=30, stackoverflow=45)[dataset]
        args.gradient_checkpointing = gradient_checkpointing
        args.last_layer_checkpointing = last_layer_checkpointing
        args.memory_safe_loop = memory_safe_loop
        args.recompute_feature_graphs = recompute_feature_graphs
        dump(folder / 'config_lock.json', vars(args))
        dataloader.set_seed(seed)
        data = object.__new__(dataloader.Data)
        data.known_label_list = known
        data.n_known_cls = data.num_labels = len(known)
        data.unseen_token = '__oos__'
        data.unseen_token_id = len(known)
        data.label_list = known + ['__oos__']
        def examples(split):
            rows = json.loads((data_root / 'gate' / f'{split}.json').read_text())
            if split != 'test':
                assert all(r['label'] == 0 and r['intent'] in known for r in rows)
            return [dataloader.InputExample(f'{split}-{i}', r['text'], label=(r['intent']
                    if r['intent'] in known else '__oos__')) for i, r in enumerate(rows)]
        data.train_examples = examples('train')
        data.eval_examples = examples('val')
        data.train_dataloader = data.get_loader(data.train_examples, args, 'train')
        data.eval_dataloader = data.get_loader(data.eval_examples, args, 'eval')
        manager = PretrainModelManager(args, data)
        if manager.device.type != 'cuda':
            raise RuntimeError('MOGB selected CPU despite required CUDA execution')
        if train_batch_size is not None or eval_batch_size is not None:
            manifest['modifications'].append(
                'compatibility-only dataloader batch-size override; official model, '
                'loss, clustering and boundary logic unchanged')
        if recompute_feature_graphs:
            os.environ['MOGB_MINIMAL_HIDDEN_STATES'] = '1'
            manifest['modifications'].append(
                'compatibility-only feature-graph recomputation with post-CE '
                'parameter/RNG replay; official mean cluster loss and one optimizer2 step preserved')
            manifest['modifications'].append(
                'compatibility-only last-hidden-state path; unused BERT hidden-state '
                'tuple is not materialized, final representation unchanged')
        if gradient_checkpointing:
            manager.model.bert.gradient_checkpointing_enable({'use_reentrant': False})
            manifest['modifications'].append(
                'non-reentrant BERT activation checkpointing enabled for memory-safe compatibility')
            manifest['gradient_checkpointing'] = True
        else:
            manifest['gradient_checkpointing'] = False
        manifest['last_layer_checkpointing'] = last_layer_checkpointing
        manifest['memory_safe_loop'] = memory_safe_loop
        if last_layer_checkpointing:
            manifest['modifications'].append(
                'non-reentrant checkpointing enabled only for trainable BERT layer 11')
        manifest['cuda_cache_cleanup_after_eval'] = True
        manifest['modifications'].append(
            'unused CUDA allocator cache released after each Known-dev evaluation')
        manifest.update(pid=os.getpid(), started_at=time.time(), device=str(manager.device))
        dump(folder / 'MANIFEST.json', manifest)
        history = []
        original_eval = manager.eval
        def logged_eval(args, data):
            score = original_eval(args, data)
            if not all(torch.isfinite(p).all() for p in manager.model.parameters()):
                raise RuntimeError('Official training produced non-finite parameters; core loss unchanged')
            gc.collect()
            torch.cuda.empty_cache()
            history.append(dict(epoch=len(history)+1, known_dev_accuracy=score,
                                cuda_allocated_bytes=int(torch.cuda.memory_allocated()),
                                cuda_reserved_bytes=int(torch.cuda.memory_reserved())))
            dump(folder / 'history.json', history)
            return score
        manager.eval = logged_eval
        train_cleanup = None
        if recompute_feature_graphs:
            import pretrain as mogb_pretrain
            train = lambda train_args, train_data: _recompute_feature_graph_official_train(
                manager, train_args, train_data, mogb_pretrain)
            manifest['training_loop'] = (
                'official CE/cluster objective with compatibility-only '
                'feature-graph recomputation')
        elif memory_safe_loop:
            import pretrain as mogb_pretrain
            train = lambda train_args, train_data: _memory_safe_official_train(
                manager, train_args, train_data, mogb_pretrain,
                epoch_cleanup=train_cleanup)
            manifest['training_loop'] = (
                'official PretrainModelManager.train with compatibility-only '
                'epoch lifetime cleanup')
            manifest['modifications'].append(
                'official loop lifetime cleanup prevents cross-epoch graph retention')
        else:
            train = manager.train
        # Persist the actual compatibility loop before training starts.  The
        # initial manifest is written before the official imports and therefore
        # only contains the requested flags.
        dump(folder / 'MANIFEST.json', manifest)
        # Official code retains each minibatch feature graph until epoch end.
        # CPU snapshots preserve those exact feature graphs across optimizer steps.
        with (_last_layer_checkpointing(manager.model)
              if last_layer_checkpointing else contextlib.nullcontext()):
            if recompute_feature_graphs:
                # This path deliberately has no saved-tensor hook: its first
                # feature pass is no-grad and its second pass releases each
                # graph immediately after backward.
                train(args, data)
            elif offload == 'cpu':
                with torch.autograd.graph.save_on_cpu(pin_memory=False):
                    train(args, data)
            elif offload == 'cpu-feature':
                with _feature_only_saved_tensors(manager.model, CpuSavedTensor):
                    train(args, data)
            elif offload == 'cpu-fp16':
                with _feature_only_saved_tensors(manager.model, CpuFp16SavedTensor):
                    train(args, data)
            elif offload == 'cpu-int8':
                with _feature_only_saved_tensors(manager.model, CpuInt8SavedTensor):
                    train(args, data)
            elif offload == 'direct':
                with tempfile.TemporaryDirectory(prefix='autograd_', dir=folder) as scratch:
                    with DirectTensorStore(scratch) as store:
                        if memory_safe_loop:
                            train_cleanup = store.reset_epoch
                        with _feature_only_saved_tensors(
                                manager.model, lambda value: DirectSavedTensor(value, store)):
                            train(args, data)
            else:
                # Disk offload is needed for full CLINC batches on the 16GB host.
                # Each object owns only its generated temporary tensor file.
                with tempfile.TemporaryDirectory(prefix='autograd_', dir=folder) as scratch:
                    def drop_file_cache(path):
                        # Linux writeback cache is charged to the service cgroup.
                        # Release clean tensor pages after each save/load so disk
                        # offload does not become a second RAM buffer.
                        advise = getattr(os, 'POSIX_FADV_DONTNEED', None)
                        if advise is None:
                            return
                        fd = os.open(path, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, advise)
                        finally:
                            os.close(fd)

                    class SavedTensor:
                        def __init__(self, tensor):
                            self.device = tensor.device
                            handle = tempfile.NamedTemporaryFile(dir=scratch, suffix='.npy', delete=False)
                            self.path = Path(handle.name)
                            with handle:
                                np.save(handle, tensor.detach().cpu().numpy())
                                handle.flush()
                                os.fsync(handle.fileno())
                                advise = getattr(os, 'POSIX_FADV_DONTNEED', None)
                                if advise is not None:
                                    os.posix_fadvise(handle.fileno(), 0, 0, advise)
                        def unpack(self):
                            value = np.load(self.path)
                            drop_file_cache(self.path)
                            return torch.from_numpy(value).to(self.device)
                        def __del__(self):
                            self.path.unlink(missing_ok=True)
                    with torch.autograd.graph.saved_tensors_hooks(SavedTensor, lambda value: value.unpack()):
                        train(args, data)
        with torch.no_grad():
            centers, radii, labels = manager.calculate_granular_balls(args, data)
        if not len(radii) or not torch.isfinite(centers).all() or not torch.isfinite(radii).all():
            raise RuntimeError('Official final balls are empty or non-finite')
        torch.save(dict(centers=centers.cpu(), radii=radii.cpu(), labels=labels.cpu()), folder / 'balls.pt')
        manifest.update(status='locked', ball_count=len(radii), test_read=False)
        dump(folder / 'MANIFEST.json', manifest)
        # Test is first materialized here, after model and ball state are locked.
        data.test_examples = examples('test')
        data.test_dataloader = data.get_loader(data.test_examples, args, 'test')
        evaluator = ModelManager(args, data, manager.model)
        manager.model.eval()
        gold, pred, scores = [], [], []
        with torch.no_grad():
            for batch in data.test_dataloader:
                ids, mask, segment, truth = (x.to(manager.device) for x in batch)
                features, _ = manager.model(ids, segment, mask)
                predicted = evaluator.open_classify(data, features, centers, radii, labels)
                distances = torch.cdist(features, centers)
                nearest = distances.argmin(dim=1)
                scores.extend((distances.gather(1, nearest[:,None]).flatten() / radii[nearest]).cpu().tolist())
                gold.extend(data.label_list[i] for i in truth.cpu().tolist())
                pred.extend(data.label_list[i] for i in predicted.cpu().tolist())
        np.savez_compressed(folder / 'predictions.npz', y_true=np.asarray(gold), y_pred=np.asarray(pred), score=scores)
        metrics = prediction_metrics(gold, pred, known)
        dump(folder / 'result.json', dict(dataset=dataset, kir=kir, seed=seed,
            method='MOGB-official-compatible', metrics=metrics, full_pipeline=False,
            task='end_to_end_open_intent_classifier', evaluator='common final prediction metrics',
            predictions=str(folder / 'predictions.npz'), manifest=str(folder / 'MANIFEST.json')))
        manifest.update(status='complete', test_read=True)
        dump(folder / 'MANIFEST.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', failure=repr(error), traceback=traceback.format_exc())
        dump(folder / 'MANIFEST.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['banking77','clinc150','stackoverflow'])
    parser.add_argument('--kir', type=float, choices=[.25,.5,.75])
    parser.add_argument('--seed', type=int, choices=[13,42,87])
    parser.add_argument('--offload', choices=['cpu','cpu-feature','cpu-fp16','cpu-int8','disk','direct'], default='disk')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--last-layer-checkpointing', action='store_true')
    parser.add_argument('--memory-safe-loop', action='store_true')
    parser.add_argument('--train-batch-size', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--recompute-feature-graphs', action='store_true')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='root containing dataset/kir_seed/gate views')
    parser.add_argument('--data-dataset', default=None,
                        help='directory name when it differs from the reported dataset')
    parser.add_argument('--output-root', type=Path, default=None,
                        help='separate output root for this matrix')
    parser.add_argument('--model-root', type=Path, default=None,
                        help='local BERT model directory')
    parser.add_argument('--matrix', action='store_true')
    a = parser.parse_args()
    if not a.matrix:
        if a.dataset is None or a.kir is None or a.seed is None:
            parser.error('single run requires dataset, kir and seed')
        run(a.dataset, a.kir, a.seed, a.offload, a.gradient_checkpointing,
            a.last_layer_checkpointing, a.memory_safe_loop,
            a.train_batch_size, a.eval_batch_size,
            a.recompute_feature_graphs, a.data_root, a.output_root, a.data_dataset, a.model_root)
    else:
        plan = [dict(dataset=d,kir=k,seed=s) for d in ('banking77','stackoverflow','clinc150')
                for k in (.25,.5,.75) for s in (13,42,87)]
        output_base = Path(a.output_root) if a.output_root is not None else OUT
        output_base.mkdir(parents=True, exist_ok=True)
        dump(output_base/'matrix_contract.json',dict(cells=plan,offload=a.offload,
             gradient_checkpointing=a.gradient_checkpointing,
             last_layer_checkpointing=a.last_layer_checkpointing,
             memory_safe_loop=a.memory_safe_loop,
             train_batch_size=a.train_batch_size,
             eval_batch_size=a.eval_batch_size,
             recompute_feature_graphs=a.recompute_feature_graphs,
             data_root=str(a.data_root) if a.data_root is not None else str(ART / 'data'),
             output_root=str(output_base),
             official_defaults=True,test_used_for_selection=False))
        for cell in plan:
            folder = output_base/cell['dataset']/f"kir{round(cell['kir']*100):02d}_seed{cell['seed']}"
            if (folder/'result.json').exists():
                continue
            folder.mkdir(parents=True,exist_ok=True)
            command = [sys.executable,str(Path(__file__).resolve()),'--dataset',cell['dataset'],
                       '--kir',str(cell['kir']),'--seed',str(cell['seed']),'--offload',a.offload]
            if a.gradient_checkpointing:
                command.append('--gradient-checkpointing')
            if a.last_layer_checkpointing:
                command.append('--last-layer-checkpointing')
            if a.memory_safe_loop:
                command.append('--memory-safe-loop')
            if a.train_batch_size is not None:
                command.extend(['--train-batch-size', str(a.train_batch_size)])
            if a.eval_batch_size is not None:
                command.extend(['--eval-batch-size', str(a.eval_batch_size)])
            if a.recompute_feature_graphs:
                command.append('--recompute-feature-graphs')
            if a.data_root is not None:
                command.extend(['--data-root', str(a.data_root)])
            if a.output_root is not None:
                command.extend(['--output-root', str(a.output_root)])
            dump(output_base/'matrix_status.json',dict(status='running',current=cell))
            with (folder/'training.log').open('a') as log:
                child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                while child.poll() is None:
                    dump(output_base/'matrix_status.json',dict(status='running',current=cell,
                         pid=child.pid, heartbeat=time.time()))
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            print(json.dumps(dict(cell=cell,returncode=child.returncode)),flush=True)
            result_path = folder/'result.json'
            if child.returncode or not result_path.exists():
                manifest_path = folder/'MANIFEST.json'
                failed = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                failed.update(status='failed', return_code=child.returncode,
                              failure=failed.get('failure',
                                      'child exited without completed result'))
                dump(manifest_path,failed)
                dump(output_base/'matrix_status.json',dict(status='failed',current=cell,
                     return_code=child.returncode, result_exists=False,
                     heartbeat=time.time()))
                raise SystemExit(child.returncode if child.returncode else 1)
            subprocess.run([sys.executable,str(ROOT/'tools/analysis/build_final_paper_main.py')],check=True)
        finished = sum((output_base/c['dataset']/f"kir{round(c['kir']*100):02d}_seed{c['seed']}"/'result.json').exists() for c in plan)
        dump(output_base/'matrix_status.json',dict(status='complete' if finished==27 else 'completed_with_failures',
                                                   completed=finished,planned=27))
