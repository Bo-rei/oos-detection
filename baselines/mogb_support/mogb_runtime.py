#!/usr/bin/env python3
"""Runtime repair launcher for the pinned official MOGB entrypoint.

This wrapper keeps the upstream checkout read-only. It monkeypatches the
official ``PretrainModelManager`` training path to avoid stale-graph failures
under modern PyTorch, then executes the untouched official ``MOGB.py`` via
``runpy``.
"""

from __future__ import annotations

import argparse
import copy
import runpy
import sys
from pathlib import Path

import torch
from tqdm import tqdm, trange


def _parse_runtime_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", type=Path, required=True)
    parser.add_argument("--compat-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--enable-anomaly-detect", action="store_true")
    return parser.parse_known_args(argv)


def _prepare_paths(args: argparse.Namespace) -> None:
    official_script = args.official_script.resolve()
    compat_root = args.compat_root.resolve()
    official_root = (args.official_root or official_script.parent).resolve()
    for path in (str(compat_root), str(official_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    args.official_script = official_script
    args.compat_root = compat_root
    args.official_root = official_root


def _collect_train_features(manager, data, *, training_mode: bool) -> tuple[torch.Tensor, torch.Tensor]:
    previous_mode = manager.model.training
    manager.model.train(training_mode)
    feature_bank = []
    label_bank = []
    for batch in tqdm(data.train_dataloader, desc="FeatureBank", leave=False):
        batch = tuple(t.to(manager.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            features = manager.model(input_ids, segment_ids, input_mask, feature_ext=True)
        feature_bank.append(features.detach().cpu())
        label_bank.append(label_ids.detach().cpu())
    if not feature_bank:
        manager.model.train(previous_mode)
        return (
            torch.empty((0, manager.clusterLoss.feat_dim), dtype=torch.float32, device=manager.device),
            torch.empty((0,), dtype=torch.long, device=manager.device),
        )
    features = torch.cat(feature_bank, dim=0).to(manager.device)
    labels = torch.cat(label_bank, dim=0).to(manager.device)
    manager.model.train(previous_mode)
    return features, labels


def _compute_fixed_ball_loss(manager, data, centroids: torch.Tensor, centroid_labels: torch.Tensor) -> torch.Tensor:
    total_items = max(len(data.train_examples), 1)
    manager.model.train()
    manager.optimizer2.zero_grad()
    accumulated = torch.tensor(0.0, device=manager.device)
    for batch in tqdm(data.train_dataloader, desc="BallUpdate", leave=False):
        batch = tuple(t.to(manager.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        features = manager.model(input_ids, segment_ids, input_mask, feature_ext=True)
        batch_loss = manager.clusterLoss.compute_classification_loss(
            features,
            label_ids,
            centroids,
            centroid_labels,
        )
        weight = input_ids.size(0) / float(total_items)
        weighted_loss = batch_loss * weight
        weighted_loss.backward()
        accumulated = accumulated + weighted_loss.detach()
    manager.optimizer2.step()
    return accumulated


def _patched_train(self, args, data):
    wait = 0
    best_model = None
    best_balls = None
    total_batches = len(data.train_dataloader)
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        self.model.train()
        tr_loss = 0.0
        nb_tr_steps = 0

        for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(True):
                loss1 = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")
                self.optimizer.zero_grad()
                loss1.backward()
                self.optimizer.step()
                tr_loss += float(loss1.item())
                try:
                    from utils import util as compat_util  # type: ignore

                    compat_util.summary_writer.add_scalar("Loss/loss1", loss1.item(), step + epoch * total_batches)
                except Exception:
                    pass
                nb_tr_steps += 1

        accumulated_features, accumulated_labels = _collect_train_features(self, data, training_mode=True)
        gb_centroids, gb_radii, gb_labels, _ = self.clusterLoss.forward(
            args,
            accumulated_features,
            accumulated_labels,
            select=False,
        )
        self.gb_centroids = gb_centroids
        self.gb_radii = gb_radii
        self.gb_labels = gb_labels

        loss2 = _compute_fixed_ball_loss(
            self,
            data,
            self.gb_centroids.detach(),
            self.gb_labels.detach(),
        )
        try:
            from utils import util as compat_util  # type: ignore

            compat_util.summary_writer.add_scalar("Loss/loss11", float(loss2.item()), epoch * total_batches)
        except Exception:
            pass

        loss = tr_loss / max(nb_tr_steps, 1)
        print("train_loss", loss)
        eval_score = self.eval(args, data)
        print("eval_score", eval_score)
        if eval_score > self.best_eval_score:
            best_model = copy.deepcopy(self.model)
            best_balls = (
                self.gb_centroids.detach().cpu().clone(),
                self.gb_radii.detach().cpu().clone(),
                self.gb_labels.detach().cpu().clone(),
            )
            wait = 0
            self.best_eval_score = eval_score
        else:
            wait += 1
            if wait >= args.wait_patient:
                break

    if best_model is not None:
        self.model = best_model
    if best_balls is not None:
        self.gb_centroids = best_balls[0].to(self.device)
        self.gb_radii = best_balls[1].to(self.device)
        self.gb_labels = best_balls[2].to(self.device)
    if args.save_model:
        self.save_model(args)
    return self.gb_centroids, self.gb_radii, self.gb_labels


def _patched_calculate_granular_balls(self, args, data):
    accumulated_features, accumulated_labels = _collect_train_features(self, data, training_mode=False)
    gb_centroids, gb_radii, gb_labels, _ = self.clusterLoss.forward(
        args,
        accumulated_features,
        accumulated_labels,
        select=True,
    )
    self.gb_centroids = gb_centroids
    self.gb_radii = gb_radii
    self.gb_labels = gb_labels
    return self.gb_centroids, self.gb_radii, self.gb_labels


def _patched_cluster_forward(self, args, features, labels, select):
    """Keep the upstream ball construction while honoring the active device.

    ``cluster.py`` hard-codes ``cuda:0`` for the sample-index column.  That is
    harmless on the author's single-GPU setup but prevents a CPU smoke (and
    any non-zero CUDA device) from reaching the otherwise unchanged GBNR
    implementation.  The tensor layout and GBNR call are intentionally kept
    identical to the upstream method.
    """

    import cluster  # type: ignore

    a_purity = args.purity_train if not select else args.purity_get_ball
    original_target = labels
    index1 = torch.arange(len(labels), device=self.device)
    label_features = torch.cat((labels.reshape(-1, 1), features), dim=1)
    out = torch.cat((index1.reshape(-1, 1), label_features), dim=1)
    out = torch.cat((original_target.reshape(-1, 1), out), dim=1)
    purity = torch.full((out.size(0), 1), float(a_purity), device=self.device)
    out = torch.cat((purity, out), dim=1)
    self.center, self.labels, self.radius = cluster.GBNR.apply(args, out.to(self.device), select)
    return self.center, self.radius, self.labels


def _apply_runtime_patch() -> None:
    import pretrain  # type: ignore
    import cluster  # type: ignore

    pretrain.PretrainModelManager.train = _patched_train
    pretrain.PretrainModelManager.calculate_granular_balls = _patched_calculate_granular_balls
    cluster.gbcluster.forward = _patched_cluster_forward


def main(argv: list[str] | None = None) -> int:
    runtime_args, official_argv = _parse_runtime_args(argv)
    _prepare_paths(runtime_args)
    if runtime_args.enable_anomaly_detect:
        torch.autograd.set_detect_anomaly(True)
    _apply_runtime_patch()
    sys.argv = [str(runtime_args.official_script), *official_argv]
    runpy.run_path(str(runtime_args.official_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
