"""HiLSA-MoE system pipeline: Gate -> Router -> Expert.

This module provides an end-to-end inference pipeline:
- Gate detector: multisphere OOS detector (K-means + Mahalanobis)
- Router checkpoint: SmolLM-135M + LoRA domain classifier
- Expert checkpoints: SmolLM-135M + LoRA intent classifiers per domain

Design goals:
- Keep stage responsibilities orthogonal (Gate rejects OOS only).
- Support both single-sample inference and batched evaluation.
- Reuse current training-time model wrappers for consistency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from src.router.router_model import SmolLMRouter

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelinePaths:
    """Path bundle for loading the system pipeline."""

    model_path: Path
    gate_encoder_path: Path
    gate_detector_path: Path
    router_ckpt_path: Path
    experts_root: Path
    experts_data_root: Path
    router_data_path: Path
    gate_train_path: Path


class HiLSAMoEV19Pipeline:
    """Hierarchical inference pipeline."""

    def __init__(
        self,
        paths: PipelinePaths,
        device: Optional[str] = None,
        max_length: int = 64,
        router_lora_r: int = 32,
        router_lora_alpha: int = 64,
        expert_lora_r: int = 16,
        expert_lora_alpha: int = 32,
        gate_radius_scale: float = 1.0,
    ) -> None:
        self.paths = paths
        self.max_length = max_length
        self.router_lora_r = router_lora_r
        self.router_lora_alpha = router_lora_alpha
        self.expert_lora_r = expert_lora_r
        self.expert_lora_alpha = expert_lora_alpha
        self.gate_radius_scale = float(gate_radius_scale)

        # In some environments, calling torch.cuda.is_available() can trigger
        # native runtime aborts. Keep initialization robust by defaulting to
        # CPU unless the caller explicitly requests a device.
        selected_device = device if device is not None else "cpu"
        self.device = torch.device(selected_device)

        self.gate_encoder: Optional[SentenceTransformer] = None
        self.gate_detector: Optional[MultiSphereOOSDetector] = None
        self.tokenizer = None
        self.router_model: Optional[SmolLMRouter] = None
        self.router_num_classes: Optional[int] = None

        self.domain_id_to_name: Dict[int, str] = {}
        self.domain_name_to_id: Dict[str, int] = {}

        self.expert_meta: Dict[str, Dict[str, Any]] = {}
        self.active_expert_domain: Optional[str] = None
        self.active_expert_model: Optional[SmolLMRouter] = None

    def load(self) -> None:
        """Load all mandatory components and metadata."""

        self._load_gate()
        self._load_tokenizer()
        self._load_router()
        self._load_domain_mapping()
        self._index_experts()

    def _load_gate(self) -> None:
        self.gate_encoder = SentenceTransformer(
            str(self.paths.gate_encoder_path),
            device=str(self.device),
        )
        self.gate_detector = MultiSphereOOSDetector()
        self.gate_detector.load(self.paths.gate_detector_path)
        self._apply_gate_radius_scale()

    def _apply_gate_radius_scale(self) -> None:
        """Apply an eval-time operating-point scale to multisphere radii."""
        if self.gate_detector is None:
            return
        if self.gate_radius_scale <= 0:
            raise ValueError("gate_radius_scale must be positive")
        if abs(self.gate_radius_scale - 1.0) < 1e-12:
            return
        for sphere in self.gate_detector.spheres:
            sphere.radius = float(sphere.radius) * self.gate_radius_scale

    def _load_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.paths.model_path), trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_router(self) -> None:
        num_classes = self._infer_router_num_classes()
        self.router_num_classes = int(num_classes)
        self.router_model = SmolLMRouter(
            model_path=str(self.paths.model_path),
            num_classes=int(num_classes),
            lora_r=self.router_lora_r,
            lora_alpha=self.router_lora_alpha,
        ).to(self.device)
        state_dict = torch.load(self.paths.router_ckpt_path, map_location=self.device)
        self.router_model.load_state_dict(state_dict)
        self.router_model.eval()

    def _infer_router_num_classes(self) -> int:
        with open(self.paths.router_data_path, "r", encoding="utf-8") as file:
            records = json.load(file)

        labels = sorted({int(row["label"]) for row in records})
        if len(labels) == 0:
            raise RuntimeError(f"Router data is empty: {self.paths.router_data_path}")

        expected = list(range(len(labels)))
        if labels != expected:
            LOGGER.warning(
                "Router labels are not dense from 0..n-1 (%s); using unique-label count=%d",
                labels,
                len(labels),
            )
        return int(len(labels))

    def _load_domain_mapping(self) -> None:
        with open(self.paths.router_data_path, "r", encoding="utf-8") as file:
            records = json.load(file)

        mapping: Dict[int, str] = {}
        for row in records:
            mapping[int(row["label"])] = row["domain"]

        self.domain_id_to_name = dict(sorted(mapping.items(), key=lambda item: item[0]))
        self.domain_name_to_id = {name: idx for idx, name in self.domain_id_to_name.items()}

    def _index_experts(self) -> None:
        for domain_dir in sorted(self.paths.experts_root.iterdir()):
            if not domain_dir.is_dir():
                continue

            domain_name = domain_dir.name
            train_file = self.paths.experts_data_root / domain_name / "train.json"
            ckpt_file = domain_dir / "best_model.pt"
            if not train_file.exists() or not ckpt_file.exists():
                continue

            with open(train_file, "r", encoding="utf-8") as file:
                rows = json.load(file)

            label_to_intent: Dict[int, str] = {}
            for row in rows:
                label_to_intent[int(row["label"])] = row["intent"]

            num_classes = len(label_to_intent)
            self.expert_meta[domain_name] = {
                "checkpoint": ckpt_file,
                "num_classes": num_classes,
                "label_to_intent": label_to_intent,
            }

        if len(self.expert_meta) == 0:
            raise RuntimeError("No expert checkpoints discovered under experts_root")

    def _ensure_active_expert(self, domain_name: str) -> None:
        if domain_name == self.active_expert_domain and self.active_expert_model is not None:
            return

        if domain_name not in self.expert_meta:
            raise KeyError(f"Expert domain not found: {domain_name}")

        info = self.expert_meta[domain_name]
        model = SmolLMRouter(
            model_path=str(self.paths.model_path),
            num_classes=int(info["num_classes"]),
            lora_r=self.expert_lora_r,
            lora_alpha=self.expert_lora_alpha,
        ).to(self.device)

        state_dict = torch.load(info["checkpoint"], map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.active_expert_model = model
        self.active_expert_domain = domain_name

    @torch.no_grad()
    def _gate_predict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        assert self.gate_encoder is not None
        embeddings = self.gate_encoder.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        assert self.gate_detector is not None
        return self.gate_detector.predict_with_scores(embeddings_np)

    @torch.no_grad()
    def _router_predict(self, texts: List[str], batch_size: int = 64) -> Dict[str, Any]:
        assert self.router_model is not None and self.tokenizer is not None

        all_domain_ids: List[int] = []
        all_domain_probs: List[float] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            logits = self.router_model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

            all_domain_ids.extend(pred.detach().cpu().tolist())
            all_domain_probs.extend(conf.detach().cpu().tolist())

        return {
            "domain_ids": all_domain_ids,
            "domain_probs": all_domain_probs,
        }

    @torch.no_grad()
    def _expert_predict_group(
        self,
        domain_name: str,
        texts: List[str],
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        assert self.tokenizer is not None
        self._ensure_active_expert(domain_name)
        assert self.active_expert_model is not None

        all_intent_ids: List[int] = []
        all_intent_probs: List[float] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            logits = self.active_expert_model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

            all_intent_ids.extend(pred.detach().cpu().tolist())
            all_intent_probs.extend(conf.detach().cpu().tolist())

        label_to_intent = self.expert_meta[domain_name]["label_to_intent"]
        all_intent_names = [label_to_intent[int(idx)] for idx in all_intent_ids]

        return {
            "intent_ids": all_intent_ids,
            "intent_probs": all_intent_probs,
            "intent_names": all_intent_names,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        """Run Gate -> Router -> Expert on a text batch."""
        if len(texts) == 0:
            return []

        gate_out = self._gate_predict(texts)
        gate_preds = gate_out["pred"]

        results: List[Dict[str, Any]] = [
            {
                "text": text,
                "is_oos": bool(gate == 1),
                "gate_pred": int(gate),
                "gate_score": float(gate_out["score"][i]),
                "gate_distance": float(gate_out["distance"][i]),
                "gate_radius": float(gate_out["radius"][i]),
                "gate_margin_ok": bool(gate_out["margin_ok"][i]),
                "gate_nearest_cluster": int(gate_out["nearest_cluster"][i]),
                "gate_nearest_intent": self.gate_detector.cluster_to_intent.get(
                    int(gate_out["nearest_cluster"][i])
                ),
            }
            for i, (text, gate) in enumerate(zip(texts, gate_preds))
        ]

        known_indices = [idx for idx, gate in enumerate(gate_preds.tolist()) if gate == 0]
        if len(known_indices) == 0:
            return results

        known_texts = [texts[idx] for idx in known_indices]
        router_out = self._router_predict(known_texts, batch_size=batch_size)

        predicted_domains = [
            self.domain_id_to_name[int(domain_id)] for domain_id in router_out["domain_ids"]
        ]

        for local_idx, global_idx in enumerate(known_indices):
            results[global_idx]["domain_id"] = int(router_out["domain_ids"][local_idx])
            results[global_idx]["domain"] = predicted_domains[local_idx]
            results[global_idx]["domain_prob"] = float(router_out["domain_probs"][local_idx])

        domain_to_local_indices: Dict[str, List[int]] = {}
        for local_idx, domain_name in enumerate(predicted_domains):
            domain_to_local_indices.setdefault(domain_name, []).append(local_idx)

        for domain_name, local_indices in domain_to_local_indices.items():
            group_texts = [known_texts[i] for i in local_indices]
            expert_out = self._expert_predict_group(
                domain_name=domain_name,
                texts=group_texts,
                batch_size=batch_size,
            )

            for group_pos, local_idx in enumerate(local_indices):
                global_idx = known_indices[local_idx]
                results[global_idx]["intent_id"] = int(expert_out["intent_ids"][group_pos])
                results[global_idx]["intent"] = expert_out["intent_names"][group_pos]
                results[global_idx]["intent_prob"] = float(expert_out["intent_probs"][group_pos])

        return results

    def predict_one(self, text: str) -> Dict[str, Any]:
        """Convenience wrapper for single text prediction."""
        return self.predict_batch([text])[0]
