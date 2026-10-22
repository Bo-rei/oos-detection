"""HiLSA-MoE v19 system pipeline: Gate -> Router -> Expert.

This module provides an end-to-end inference pipeline using existing v19
artifacts:
- Gate detector: outputs/multisphere_corrected/corrected_multisphere_detector.json
- Router checkpoint: outputs/router_v19/best_model.pt
- Expert checkpoints: outputs/experts_v19/<domain>/best_model.pt

Design goals:
- Keep stage responsibilities orthogonal (Gate rejects OOS only).
- Support both single-sample inference and batched evaluation.
- Reuse current training-time model wrappers for consistency.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from src.gate.intent_prototype_matcher import IntentPrototypeMatcher
from src.gate.llm_semantic_verifier import LLMSemanticVerifier, LLMVerifierConfig
from src.gate.multi_prototype_gate import MultiPrototypeGate
from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from src.router import SmolLMRouter

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelinePaths:
    """Path bundle for loading the v19 system pipeline."""

    model_path: Path
    gate_encoder_path: Path
    gate_detector_path: Path
    router_ckpt_path: Path
    experts_root: Path
    experts_data_root: Path
    router_data_path: Path
    gate_train_path: Path
    # 表示适配实验可选的 AutoModel checkpoint。为空时保持历史
    # SentenceTransformer 加载路径，避免把两种 checkpoint 格式混用。
    gate_encoder_checkpoint_path: Optional[Path] = None
    # validation 选出的线性置信度 Gate（MSP/Entropy/Energy）序列化文件。
    gate_baseline_path: Optional[Path] = None
    prompt_semantic_verifier_ckpt_path: Optional[Path] = None
    semantic_verifier_ckpt_path: Optional[Path] = None
    multi_prototype_path: Optional[Path] = None


class _AdaptedSentenceEncoder:
    """把表示适配训练输出接入 Gate 所需的 ``encode`` 接口。

    CE/SupCon/CE-Recon 训练脚本保存的是 ``AutoModel.state_dict``，不是
    sentence-transformers 目录。这里复用训练时的 mean pooling 和截断长度，
    确保端到端评价使用与表示实验一致的语义空间。
    """

    def __init__(self, model_path: Path, checkpoint_path: Path, device: torch.device) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(payload["encoder"])
        self.model.to(device)
        self.model.eval()
        self.device = device

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), int(batch_size)):
            batch = self.tokenizer(
                list(texts[start : start + int(batch_size)]),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            pooled = self._mean_pool(self.model(**batch).last_hidden_state, batch["attention_mask"])
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        if outputs:
            return np.concatenate(outputs, axis=0)
        return np.empty((0, int(self.model.config.hidden_size)), dtype=np.float32)


class HiLSAMoEV19Pipeline:
    """Hierarchical inference pipeline for v19."""

    def __init__(
        self,
        paths: PipelinePaths,
        device: Optional[str] = None,
        max_length: int = 64,
        router_lora_r: int = 32,
        router_lora_alpha: int = 64,
        expert_lora_r: int = 16,
        expert_lora_alpha: int = 32,
        semantic_gate_enabled: bool = False,
        semantic_gate_mode: str = "prototype",
        semantic_uncertain_low: float = 0.98,
        semantic_uncertain_high: float = 1.05,
        semantic_gate_threshold: float = 0.45,
        semantic_top_k: int = 3,
        semantic_prompt_version: str = "ranking_v1",
        prototype_centers_default: int = 1,
        prototype_centers_overrides: Optional[Dict[str, int]] = None,
        semantic_fusion_alpha: float = 0.7,
        semantic_fusion_beta: float = 0.3,
        semantic_decision_policy: str = "threshold",
        semantic_low_conf_threshold: float = 0.80,
        semantic_high_conf_threshold: float = 0.90,
        semantic_verifier_threshold: float = 0.50,
        semantic_verifier_lora_r: int = 16,
        semantic_verifier_lora_alpha: int = 32,
        gate_mode: str = "multisphere",
        multi_proto_id_threshold: float = 0.80,
        gate_radius_scale: float = 1.0,
    ) -> None:
        self.paths = paths
        self.max_length = max_length
        self.router_lora_r = router_lora_r
        self.router_lora_alpha = router_lora_alpha
        self.expert_lora_r = expert_lora_r
        self.expert_lora_alpha = expert_lora_alpha
        self.semantic_gate_enabled = semantic_gate_enabled
        self.semantic_gate_mode = semantic_gate_mode
        self.semantic_uncertain_low = semantic_uncertain_low
        self.semantic_uncertain_high = semantic_uncertain_high
        self.semantic_gate_threshold = semantic_gate_threshold
        self.semantic_top_k = int(max(1, semantic_top_k))
        self.semantic_prompt_version = str(semantic_prompt_version)
        self.semantic_fusion_alpha = float(semantic_fusion_alpha)
        self.semantic_fusion_beta = float(semantic_fusion_beta)
        self.semantic_decision_policy = str(semantic_decision_policy)
        self.semantic_low_conf_threshold = float(semantic_low_conf_threshold)
        self.semantic_high_conf_threshold = float(semantic_high_conf_threshold)
        self.semantic_verifier_threshold = float(semantic_verifier_threshold)
        self.semantic_verifier_lora_r = int(semantic_verifier_lora_r)
        self.semantic_verifier_lora_alpha = int(semantic_verifier_lora_alpha)
        self.gate_mode = str(gate_mode)
        self.multi_proto_id_threshold = float(multi_proto_id_threshold)
        self.gate_radius_scale = float(gate_radius_scale)
        self.prototype_centers_default = int(max(1, prototype_centers_default))
        self.prototype_centers_overrides = prototype_centers_overrides or {}

        # In some environments, calling torch.cuda.is_available() can trigger
        # native runtime aborts. Keep initialization robust by defaulting to
        # CPU unless the caller explicitly requests a device.
        selected_device = device if device is not None else "cpu"
        self.device = torch.device(selected_device)

        self.gate_encoder: Optional[SentenceTransformer] = None
        self.gate_detector: Optional[MultiSphereOOSDetector] = None
        self.gate_baseline: Optional[Dict[str, Any]] = None
        self.multi_prototype_gate: Optional[MultiPrototypeGate] = None
        self.multi_prototype_intent_to_domain: Dict[str, str] = {}
        self.tokenizer = None
        self.router_model: Optional[SmolLMRouter] = None
        self.router_num_classes: Optional[int] = None

        self.domain_id_to_name: Dict[int, str] = {}
        self.domain_name_to_id: Dict[str, int] = {}

        self.expert_meta: Dict[str, Dict[str, Any]] = {}
        self.active_expert_domain: Optional[str] = None
        self.active_expert_model: Optional[SmolLMRouter] = None
        self.prototype_matcher: Optional[IntentPrototypeMatcher] = None
        self.llm_semantic_verifier: Optional[LLMSemanticVerifier] = None
        self.prompt_semantic_verifier_model: Optional[SmolLMRouter] = None
        self.binary_semantic_verifier: Optional[SmolLMRouter] = None

    def route_uncertain_gate_samples(self, gate_scores: np.ndarray) -> List[int]:
        """Select uncertain gate samples by normalized distance score interval."""
        uncertain_indices: List[int] = []
        for idx, score in enumerate(gate_scores.tolist()):
            if self.semantic_uncertain_low <= float(score) <= self.semantic_uncertain_high:
                uncertain_indices.append(idx)
        return uncertain_indices

    @torch.no_grad()
    def smollm_semantic_gate_verify(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Use SmolLM prototype similarity and optional decoder-only reranking."""
        if len(texts) == 0:
            return {
                "semantic_id_scores": [],
                "semantic_decision_scores": [],
                "semantic_is_id": [],
                "semantic_top_intent": [],
                "semantic_top_domain": [],
                "semantic_candidate_best_intent": [],
                "semantic_candidate_best_score": [],
                "semantic_candidate_runner_up_intent": [],
                "semantic_candidate_runner_up_score": [],
                "semantic_candidate_score_margin": [],
                "semantic_mode": self.semantic_gate_mode,
            }
        if self.prototype_matcher is None:
            raise RuntimeError("Prototype matcher is not initialized for semantic gate.")

        proto_out = self.prototype_matcher.score_texts_to_prototypes(texts)
        semantic_scores = [float(score) for score in proto_out["max_similarity"]]
        semantic_decision_scores = list(semantic_scores)
        semantic_ranking: List[List[str]] = []
        semantic_ranking_scores: List[List[float]] = []
        semantic_candidate_best_intents: List[str] = []
        semantic_candidate_best_scores: List[float] = []
        semantic_candidate_runner_up_intents: List[str] = []
        semantic_candidate_runner_up_scores: List[float] = []
        semantic_candidate_score_margins: List[float] = []

        if self.semantic_gate_mode in {"llm_verifier", "fusion"}:
            if self.llm_semantic_verifier is None:
                raise RuntimeError("LLM semantic verifier is not initialized.")
            verifier_scores: List[float] = []
            for text in texts:
                if self.prototype_matcher is None:
                    raise RuntimeError("Prototype matcher is not initialized for semantic gate.")
                top_k_pack = self.prototype_matcher.top_k_intents(str(text), top_k=self.semantic_top_k)
                candidate_pack = self.llm_semantic_verifier.rank_candidate_intents(
                    query_text=str(text),
                    candidate_intents=top_k_pack["candidate_intents"],
                )
                semantic_ranking.append(candidate_pack["candidate_intents"])
                semantic_ranking_scores.append(candidate_pack["candidate_scores"])
                verifier_scores.append(float(candidate_pack["best_score"]))
                semantic_candidate_best_intents.append(str(candidate_pack["best_intent"]))
                semantic_candidate_best_scores.append(float(candidate_pack["best_score"]))
                semantic_candidate_runner_up_intents.append(str(candidate_pack["runner_up_intent"]))
                semantic_candidate_runner_up_scores.append(float(candidate_pack["runner_up_score"]))
                semantic_candidate_score_margins.append(float(candidate_pack["score_margin"]))

            if self.semantic_gate_mode == "llm_verifier":
                semantic_decision_scores = verifier_scores
            else:
                denom = max(self.semantic_fusion_alpha + self.semantic_fusion_beta, 1e-12)
                semantic_decision_scores = [
                    float(
                        (
                            self.semantic_fusion_alpha * float(proto_score)
                            + self.semantic_fusion_beta * float(verifier_score)
                        )
                        / denom
                    )
                    for proto_score, verifier_score in zip(semantic_scores, verifier_scores)
                ]
        else:
            semantic_candidate_best_intents = [str(x) for x in proto_out["top_intent"]]
            semantic_candidate_best_scores = [float(x) for x in semantic_scores]
            semantic_candidate_runner_up_intents = ["" for _ in semantic_scores]
            semantic_candidate_runner_up_scores = [float(x) for x in semantic_scores]
            semantic_candidate_score_margins = [0.0 for _ in semantic_scores]

        semantic_is_id = [score >= self.semantic_gate_threshold for score in semantic_decision_scores]

        return {
            "semantic_id_scores": semantic_scores,
            "semantic_decision_scores": [float(s) for s in semantic_decision_scores],
            "semantic_is_id": semantic_is_id,
            "semantic_top_intent": [str(x) for x in proto_out["top_intent"]],
            "semantic_top_domain": [str(x) for x in proto_out["top_domain"]],
            "semantic_candidate_ranking": semantic_ranking,
            "semantic_candidate_ranking_scores": semantic_ranking_scores,
            "semantic_candidate_best_intent": semantic_candidate_best_intents,
            "semantic_candidate_best_score": semantic_candidate_best_scores,
            "semantic_candidate_runner_up_intent": semantic_candidate_runner_up_intents,
            "semantic_candidate_runner_up_score": semantic_candidate_runner_up_scores,
            "semantic_candidate_score_margin": semantic_candidate_score_margins,
            "semantic_mode": self.semantic_gate_mode,
        }

    def load(self) -> None:
        """Load all mandatory components and metadata."""
        LOGGER.info("Loading HiLSA-MoE v19 pipeline on device=%s", self.device)

        self._load_gate()
        self._load_tokenizer()
        self._load_router()
        self._load_domain_mapping()
        self._index_experts()
        self._build_intent_prototypes()
        self._build_llm_semantic_verifier()
        self._build_binary_semantic_verifier()

    def _load_gate(self) -> None:
        if self.paths.gate_encoder_checkpoint_path is not None:
            self.gate_encoder = _AdaptedSentenceEncoder(
                self.paths.gate_encoder_path,
                self.paths.gate_encoder_checkpoint_path,
                self.device,
            )
        else:
            self.gate_encoder = SentenceTransformer(
                str(self.paths.gate_encoder_path),
                device=str(self.device),
            )
        if self.gate_mode == "multisphere":
            self.gate_detector = MultiSphereOOSDetector()
            self.gate_detector.load(self.paths.gate_detector_path)
            self._apply_gate_radius_scale()
            return

        if self.gate_mode == "multi_prototype":
            if self.paths.multi_prototype_path is None:
                raise ValueError("gate_mode=multi_prototype requires multi_prototype_path")

            with open(self.paths.multi_prototype_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            if "multi_prototype_gate" in payload:
                gate_state = payload["multi_prototype_gate"]
                intent_to_domain = payload.get("intent_to_domain", {})
            elif "prototypes" in payload:
                prototypes_map = payload.get("prototypes", {})
                gate_state = {
                    "config": {
                        "k_default": int(payload.get("default_centers", 1)),
                        "k_overrides": {str(k): int(v) for k, v in payload.get("overrides", {}).items()},
                        "random_state": int(payload.get("random_state", 42)),
                        "n_init": int(payload.get("n_init", 10)),
                        "l2_normalize": True,
                    },
                    "intent_labels": sorted([str(intent) for intent in prototypes_map.keys()]),
                    "prototypes": {
                        str(intent): value.get("centers", [])
                        for intent, value in prototypes_map.items()
                    },
                    "is_fitted": True,
                }
                intent_to_domain = {
                    str(intent): str(value.get("domain", "unknown"))
                    for intent, value in prototypes_map.items()
                }
            else:
                raise ValueError("Invalid multi_prototype payload format")

            self.multi_prototype_gate = MultiPrototypeGate.from_dict(gate_state)
            # Apply runtime center-count ablation controls to loaded prototypes.
            # This ensures k=1 / k=5 style studies actually differ when loading
            # a prebuilt multi_prototype payload.
            if self.multi_prototype_gate is not None:
                adjusted: Dict[str, np.ndarray] = {}
                for intent, centers in self.multi_prototype_gate.prototypes.items():
                    desired_k = int(
                        max(
                            1,
                            self.prototype_centers_overrides.get(
                                str(intent), self.prototype_centers_default
                            ),
                        )
                    )
                    effective_k = int(min(desired_k, int(centers.shape[0])))
                    adjusted[str(intent)] = centers[:effective_k]

                self.multi_prototype_gate.prototypes = adjusted
                self.multi_prototype_gate.intent_labels = sorted(adjusted.keys())
                self.multi_prototype_gate.config.k_default = int(self.prototype_centers_default)
                self.multi_prototype_gate.config.k_overrides = {
                    str(k): int(v) for k, v in self.prototype_centers_overrides.items()
                }
            self.multi_prototype_intent_to_domain = {
                str(intent): str(domain) for intent, domain in intent_to_domain.items()
            }
            return

        if self.gate_mode == "linear_baseline":
            if self.paths.gate_baseline_path is None:
                raise ValueError("gate_mode=linear_baseline requires gate_baseline_path")
            with self.paths.gate_baseline_path.open("rb") as file:
                payload = pickle.load(file)
            if not isinstance(payload, dict) or "classifier" not in payload:
                raise ValueError("Invalid linear baseline payload")
            self.gate_baseline = payload
            return

        raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

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
        # 单域数据集（例如 BANKING77-OOS 和 StackOverflow）不存在路由
        # 决策。旧流程仍会训练一个 1-class SmolLM head，既没有额外信息，
        # 又容易在无意义的大 batch 训练中耗尽显存。这里显式采用常量路由，
        # 由 ``_router_predict`` 返回唯一 domain；多域数据集仍严格加载
        # 原有 checkpoint，保持历史 Cascade 行为不变。
        if int(num_classes) == 1:
            self.router_model = None
            LOGGER.info("Single-domain router detected; using constant routing without loading a checkpoint.")
            return
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

    def _build_intent_prototypes(self) -> None:
        if not self.semantic_gate_enabled:
            return
        assert self.router_model is not None and self.tokenizer is not None

        with open(self.paths.gate_train_path, "r", encoding="utf-8") as file:
            train_records = json.load(file)

        matcher = IntentPrototypeMatcher(
            tokenizer=self.tokenizer,
            model=self.router_model,
            device=self.device,
        )
        matcher.build_prototypes(
            records=train_records,
            default_centers=self.prototype_centers_default,
            centers_overrides=self.prototype_centers_overrides,
            random_state=42,
        )
        self.prototype_matcher = matcher

    def _build_llm_semantic_verifier(self) -> None:
        if not self.semantic_gate_enabled:
            return
        if self.semantic_gate_mode != "llm_verifier":
            if self.semantic_gate_mode != "fusion":
                return
        assert self.router_model is not None and self.tokenizer is not None
        verifier_model = self.router_model
        if self.paths.prompt_semantic_verifier_ckpt_path is not None:
            verifier_model = SmolLMRouter(
                model_path=str(self.paths.model_path),
                num_classes=2,
                lora_r=self.semantic_verifier_lora_r,
                lora_alpha=self.semantic_verifier_lora_alpha,
            ).to(self.device)
            state_dict = torch.load(
                self.paths.prompt_semantic_verifier_ckpt_path,
                map_location=self.device,
            )
            verifier_model.load_state_dict(state_dict)
            verifier_model.eval()
            self.prompt_semantic_verifier_model = verifier_model

        self.llm_semantic_verifier = LLMSemanticVerifier(
            tokenizer=self.tokenizer,
            model=verifier_model,
            device=self.device,
            config=LLMVerifierConfig(prompt_version=self.semantic_prompt_version),
        )

    def _build_binary_semantic_verifier(self) -> None:
        if not self.semantic_gate_enabled:
            return
        if self.semantic_decision_policy != "two_stage_verifier":
            return
        if self.paths.semantic_verifier_ckpt_path is None:
            raise ValueError(
                "semantic_decision_policy=two_stage_verifier requires semantic_verifier_ckpt_path"
            )

        verifier_model = SmolLMRouter(
            model_path=str(self.paths.model_path),
            num_classes=2,
            lora_r=self.semantic_verifier_lora_r,
            lora_alpha=self.semantic_verifier_lora_alpha,
        ).to(self.device)

        state_dict = torch.load(self.paths.semantic_verifier_ckpt_path, map_location=self.device)
        verifier_model.load_state_dict(state_dict)
        verifier_model.eval()
        self.binary_semantic_verifier = verifier_model

    @torch.no_grad()
    def _binary_verifier_predict(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> Dict[str, List[float]]:
        if self.binary_semantic_verifier is None:
            raise RuntimeError("Binary semantic verifier is not initialized.")
        assert self.tokenizer is not None

        prob_id_all: List[float] = []
        pred_label_all: List[int] = []

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

            logits = self.binary_semantic_verifier(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            prob_id_all.extend(probs[:, 0].detach().cpu().tolist())
            pred_label_all.extend(pred.detach().cpu().tolist())

        return {
            "prob_id": [float(value) for value in prob_id_all],
            "pred_label": [int(value) for value in pred_label_all],
        }

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

        if self.gate_mode == "multisphere":
            assert self.gate_detector is not None
            return self.gate_detector.predict_with_scores(embeddings_np)

        if self.gate_mode == "multi_prototype":
            if self.multi_prototype_gate is None:
                raise RuntimeError("MultiPrototypeGate not initialized.")

            score_and_intent = self.multi_prototype_gate.score_batch(embeddings_np)
            sim_scores = np.asarray([item[0] for item in score_and_intent], dtype=np.float32)
            best_intents = [str(item[1]) for item in score_and_intent]
            distance_like = 1.0 - sim_scores
            pred = (sim_scores < self.multi_proto_id_threshold).astype(np.int64)

            return {
                "pred": pred,
                "score": distance_like,
                "distance": distance_like,
                "radius": np.full_like(distance_like, 1.0 - self.multi_proto_id_threshold),
                "margin_ok": (pred == 0),
                "nearest_cluster": np.full_like(pred, -1),
                "nearest_intent": np.asarray(best_intents, dtype=object),
            }

        if self.gate_mode == "linear_baseline":
            if self.gate_baseline is None:
                raise RuntimeError("Linear baseline gate is not initialized")
            classifier = self.gate_baseline["classifier"]
            method = str(self.gate_baseline["method"])
            probabilities = np.asarray(classifier.predict_proba(embeddings_np), dtype=np.float64)
            logits = np.asarray(classifier.decision_function(embeddings_np), dtype=np.float64)
            if logits.ndim == 1:
                logits = np.column_stack((-0.5 * logits, 0.5 * logits))
            if method == "msp":
                scores = 1.0 - np.max(probabilities, axis=1)
            elif method == "entropy":
                clipped = np.clip(probabilities, 1e-12, 1.0)
                scores = -np.sum(clipped * np.log(clipped), axis=1) / np.log(probabilities.shape[1])
            elif method == "energy":
                shifted = logits - np.max(logits, axis=1, keepdims=True)
                scores = -(np.max(logits, axis=1) + np.log(np.exp(shifted).sum(axis=1)))
            else:
                raise ValueError(f"Unsupported linear baseline method: {method}")
            threshold = float(self.gate_baseline["threshold"])
            pred = (scores > threshold).astype(np.int64)
            nearest_intent = np.asarray(
                [str(value) for value in classifier.classes_[np.argmax(probabilities, axis=1)]],
                dtype=object,
            )
            return {
                "pred": pred,
                "score": np.asarray(scores, dtype=np.float64),
                "distance": np.asarray(scores, dtype=np.float64),
                "radius": np.full_like(scores, threshold, dtype=np.float64),
                "margin_ok": pred == 0,
                "nearest_cluster": np.full_like(pred, -1),
                "nearest_intent": nearest_intent,
            }

        raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

    @torch.no_grad()
    def _router_predict(self, texts: List[str], batch_size: int = 64) -> Dict[str, Any]:
        # 单域路径不需要 tokenizer 或模型前向；仍返回与学习型 Router
        # 完全相同的 schema，保证后续 Expert 分组无需特殊分支。
        if self.router_model is None and int(self.router_num_classes or 0) == 1:
            return {
                "domain_ids": [0 for _ in texts],
                "domain_probs": [1.0 for _ in texts],
            }

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
        final_gate_preds = np.array(gate_preds, dtype=np.int64)

        results: List[Dict[str, Any]] = [
            {
                "text": text,
                "is_oos": bool(gate == 1),
                "gate_pred": int(gate),
                "fast_gate_pred": int(gate),
                "gate_score": float(gate_out["score"][i]),
                "gate_distance": float(gate_out["distance"][i]),
                "gate_radius": float(gate_out["radius"][i]),
                "gate_margin_ok": bool(gate_out["margin_ok"][i]),
                "gate_nearest_cluster": int(gate_out["nearest_cluster"][i]),
                "gate_nearest_intent": self.gate_detector.cluster_to_intent.get(
                    int(gate_out["nearest_cluster"][i])
                )
                if self.gate_mode == "multisphere"
                else str(gate_out["nearest_intent"][i]),
                "gate_stage": "fast_gate",
                "semantic_id_score": None,
                "semantic_gate_decision": None,
                "semantic_top_intent": None,
                "semantic_top_domain": None,
                "semantic_decision_score": None,
                "semantic_candidate_best_intent": None,
                "semantic_candidate_best_score": None,
                "semantic_candidate_runner_up_intent": None,
                "semantic_candidate_runner_up_score": None,
                "semantic_candidate_score_margin": None,
                "semantic_mode": self.semantic_gate_mode,
                "semantic_policy": self.semantic_decision_policy,
                "semantic_verifier_score": None,
                "final_gate_decision": "oos" if int(gate) == 1 else "id",
            }
            for i, (text, gate) in enumerate(zip(texts, gate_preds))
        ]

        if self.semantic_gate_enabled:
            uncertain_indices = self.route_uncertain_gate_samples(gate_out["score"])
            if len(uncertain_indices) > 0:
                uncertain_texts = [texts[idx] for idx in uncertain_indices]
                semantic_out = self.smollm_semantic_gate_verify(
                    uncertain_texts,
                    batch_size=batch_size,
                )

                verifier_local_indices: List[int] = []
                verifier_texts: List[str] = []
                verifier_scores: Dict[int, float] = {}

                if self.semantic_decision_policy == "two_stage_verifier":
                    for local_idx, text in enumerate(uncertain_texts):
                        prototype_score = float(semantic_out["semantic_id_scores"][local_idx])
                        if (
                            prototype_score >= self.semantic_low_conf_threshold
                            and prototype_score <= self.semantic_high_conf_threshold
                        ):
                            verifier_local_indices.append(local_idx)
                            verifier_texts.append(text)

                    if len(verifier_texts) > 0:
                        verifier_out = self._binary_verifier_predict(
                            verifier_texts,
                            batch_size=batch_size,
                        )
                        for j, local_idx in enumerate(verifier_local_indices):
                            verifier_scores[local_idx] = float(verifier_out["prob_id"][j])

                for local_idx, global_idx in enumerate(uncertain_indices):
                    prototype_score = float(semantic_out["semantic_id_scores"][local_idx])
                    semantic_decision_score = float(semantic_out["semantic_decision_scores"][local_idx])

                    if self.semantic_decision_policy == "two_stage_verifier":
                        if prototype_score < self.semantic_low_conf_threshold:
                            semantic_is_id = False
                            gate_stage = "semantic_gate_prototype"
                        elif prototype_score > self.semantic_high_conf_threshold:
                            semantic_is_id = True
                            gate_stage = "semantic_gate_prototype"
                        else:
                            verifier_score = verifier_scores.get(local_idx)
                            if verifier_score is None:
                                verifier_score = semantic_decision_score
                            semantic_is_id = bool(verifier_score >= self.semantic_verifier_threshold)
                            gate_stage = "semantic_gate_verifier"
                            results[global_idx]["semantic_verifier_score"] = float(verifier_score)
                            semantic_decision_score = float(verifier_score)
                    else:
                        semantic_is_id = bool(semantic_out["semantic_is_id"][local_idx])
                        gate_stage = "semantic_gate"

                    final_gate_preds[global_idx] = 0 if semantic_is_id else 1

                    results[global_idx]["gate_stage"] = gate_stage
                    results[global_idx]["semantic_id_score"] = prototype_score
                    results[global_idx]["semantic_decision_score"] = semantic_decision_score
                    results[global_idx]["semantic_gate_decision"] = (
                        "id" if semantic_is_id else "oos"
                    )
                    results[global_idx]["semantic_top_intent"] = semantic_out["semantic_top_intent"][
                        local_idx
                    ]
                    results[global_idx]["semantic_top_domain"] = semantic_out["semantic_top_domain"][
                        local_idx
                    ]
                    results[global_idx]["semantic_candidate_best_intent"] = semantic_out[
                        "semantic_candidate_best_intent"
                    ][local_idx]
                    results[global_idx]["semantic_candidate_best_score"] = semantic_out[
                        "semantic_candidate_best_score"
                    ][local_idx]
                    results[global_idx]["semantic_candidate_runner_up_intent"] = semantic_out[
                        "semantic_candidate_runner_up_intent"
                    ][local_idx]
                    results[global_idx]["semantic_candidate_runner_up_score"] = semantic_out[
                        "semantic_candidate_runner_up_score"
                    ][local_idx]
                    results[global_idx]["semantic_candidate_score_margin"] = semantic_out[
                        "semantic_candidate_score_margin"
                    ][local_idx]
                    results[global_idx]["gate_pred"] = int(final_gate_preds[global_idx])
                    results[global_idx]["is_oos"] = bool(final_gate_preds[global_idx] == 1)
                    results[global_idx]["final_gate_decision"] = (
                        "id" if final_gate_preds[global_idx] == 0 else "oos"
                    )

        known_indices = [idx for idx, gate in enumerate(final_gate_preds.tolist()) if gate == 0]
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
