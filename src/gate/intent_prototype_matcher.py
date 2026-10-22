"""Intent prototype semantic matcher for hybrid open-set gate.

This module builds intent prototypes in SmolLM embedding space and provides
similarity-based semantic ID/OOS evidence for uncertain gate samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


@dataclass
class PrototypeConfig:
    """Prototype construction configuration."""

    max_length: int = 64
    batch_size: int = 64
    l2_normalize: bool = True


class IntentPrototypeMatcher:
    """Build and query intent prototypes in SmolLM embedding space."""

    def __init__(
        self,
        tokenizer: Any,
        model: torch.nn.Module,
        device: torch.device,
        config: Optional[PrototypeConfig] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.config = config or PrototypeConfig()

        self.intent_centers: Dict[str, np.ndarray] = {}
        self.intent_to_domain: Dict[str, str] = {}
        self.intent_center_counts: Dict[str, int] = {}

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text list into pooled SmolLM embeddings."""
        if len(texts) == 0:
            hidden_size = int(getattr(self.model.base.config, "hidden_size", 576))
            return np.zeros((0, hidden_size), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[-1]

            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            if self.config.l2_normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def build_prototypes(
        self,
        records: List[Dict[str, Any]],
        default_centers: int = 1,
        centers_overrides: Optional[Dict[str, int]] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Build per-intent prototype centers from training records.

        Args:
            records: Train records with keys: text, intent, domain, label.
            default_centers: Default number of centers per intent.
            centers_overrides: Optional per-intent center overrides.
        """
        centers_overrides = centers_overrides or {}

        intent_to_texts: Dict[str, List[str]] = {}
        for row in records:
            if int(row.get("label", 0)) != 0:
                continue
            intent = str(row["intent"])
            intent_to_texts.setdefault(intent, []).append(str(row["text"]))
            if intent not in self.intent_to_domain:
                self.intent_to_domain[intent] = str(row.get("domain", "unknown"))

        self.intent_centers.clear()
        self.intent_center_counts.clear()

        for intent, texts in intent_to_texts.items():
            vectors = self.embed_texts(texts)
            requested_centers = max(1, int(centers_overrides.get(intent, default_centers)))
            n_centers = min(requested_centers, max(1, vectors.shape[0]))

            if n_centers == 1:
                center = np.mean(vectors, axis=0, keepdims=True)
            else:
                kmeans = KMeans(
                    n_clusters=n_centers,
                    random_state=random_state,
                    n_init=10,
                )
                kmeans.fit(vectors)
                center = kmeans.cluster_centers_.astype(np.float32)

            if self.config.l2_normalize:
                norm = np.linalg.norm(center, axis=1, keepdims=True)
                center = center / np.clip(norm, 1e-12, None)

            self.intent_centers[intent] = center
            self.intent_center_counts[intent] = int(center.shape[0])

        return {
            "intent_count": len(self.intent_centers),
            "intent_center_counts": dict(self.intent_center_counts),
        }

    def score_texts_to_prototypes(self, texts: List[str]) -> Dict[str, Any]:
        """Compute max similarity scores against all intent prototypes."""
        if len(self.intent_centers) == 0:
            raise RuntimeError("Intent prototypes are empty. Build prototypes before scoring.")

        embeddings = self.embed_texts(texts)
        intents = sorted(self.intent_centers.keys())

        max_scores: List[float] = []
        top_intents: List[str] = []
        top_domains: List[str] = []

        for query_vec in embeddings:
            best_score = -1.0
            best_intent = ""
            for intent in intents:
                centers = self.intent_centers[intent]
                score = float(np.max(np.dot(centers, query_vec)))
                if score > best_score:
                    best_score = score
                    best_intent = intent

            max_scores.append(best_score)
            top_intents.append(best_intent)
            top_domains.append(self.intent_to_domain.get(best_intent, "unknown"))

        return {
            "max_similarity": max_scores,
            "top_intent": top_intents,
            "top_domain": top_domains,
        }

    def top_k_intents(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """Return top-k candidate intents for a single text."""
        if len(self.intent_centers) == 0:
            raise RuntimeError("Intent prototypes are empty. Build prototypes before scoring.")

        query_vec = self.embed_texts([text])[0]
        intents = sorted(self.intent_centers.keys())
        scored: List[tuple[str, float]] = []

        for intent in intents:
            centers = self.intent_centers[intent]
            score = float(np.max(np.dot(centers, query_vec)))
            scored.append((intent, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top_items = scored[: max(1, int(top_k))]
        best_intent, best_score = top_items[0]
        runner_up_intent = top_items[1][0] if len(top_items) > 1 else ""
        runner_up_score = top_items[1][1] if len(top_items) > 1 else best_score

        return {
            "candidate_intents": [item[0] for item in top_items],
            "candidate_scores": [float(item[1]) for item in top_items],
            "candidate_domains": [self.intent_to_domain.get(item[0], "unknown") for item in top_items],
            "best_intent": best_intent,
            "best_score": float(best_score),
            "runner_up_intent": runner_up_intent,
            "runner_up_score": float(runner_up_score),
            "score_margin": float(best_score - runner_up_score),
        }
