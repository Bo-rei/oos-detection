"""Prompt-based semantic verifier for uncertain open-set samples.

This verifier uses decoder-only next-token likelihood for Yes/No decisions
and supports candidate-context ranking prompts for gate reranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch


@dataclass
class LLMVerifierConfig:
    """Configuration for prompt semantic verifier."""

    max_length: int = 128
    yes_token: str = " yes"
    no_token: str = " no"
    prompt_version: str = "ranking_v1"


class LLMSemanticVerifier:
    """Compute semantic match scores with prompt-conditioned Yes/No likelihood."""

    _PROMPT_VERSIONS = {"ranking_v1", "ranking_v2", "ranking_v3"}

    def __init__(
        self,
        tokenizer,
        model,
        device: torch.device,
        config: LLMVerifierConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.config = config or LLMVerifierConfig()

        yes_ids = self.tokenizer(self.config.yes_token, add_special_tokens=False)["input_ids"]
        no_ids = self.tokenizer(self.config.no_token, add_special_tokens=False)["input_ids"]
        self.yes_token_id = int(yes_ids[0]) if yes_ids else int(self.tokenizer.eos_token_id)
        self.no_token_id = int(no_ids[0]) if no_ids else int(self.tokenizer.eos_token_id)

    @staticmethod
    def intent_to_description(intent_name: str) -> str:
        """Create a deterministic lightweight description from intent label."""
        normalized = intent_name.replace("_", " ").strip()
        return f"User intent is about: {normalized}."

    def _build_candidate_block(
        self,
        candidate_intents: Sequence[str],
        prompt_version: str,
    ) -> str:
        """Render the top-k candidate list for prompt conditioning."""
        lines = ["Candidate intents:"]
        separator = ":" if prompt_version in {"ranking_v2", "ranking_v3"} else "-"
        for idx, candidate in enumerate(candidate_intents, start=1):
            desc = self.intent_to_description(str(candidate))
            lines.append(f"{idx}. {candidate} {separator} {desc}")
        return "\n".join(lines)

    @staticmethod
    def format_prompt(
        query: str,
        intent_name: str,
        candidate_intents: Sequence[str] | None = None,
        prompt_version: str = "ranking_v1",
    ) -> str:
        """Render the shared gate prompt text for training or inference."""
        prompt_version = str(prompt_version)
        if prompt_version == "ranking_v2":
            lines = [
                "You are a strict binary verifier for intent routing.",
                "Compare the query against the target intent and the competing candidates.",
                "Answer yes only when the target intent is the best semantic match.",
                "If the query is ambiguous or fits another candidate better, answer no.",
                "Return exactly one word: yes or no.",
                f"Prompt version: {prompt_version}",
            ]
        elif prompt_version == "ranking_v3":
            lines = [
                "You are a binary verifier for intent routing.",
                "Use the candidate list to compare the target intent against nearby alternatives.",
                "Answer yes when the query is a clear or strong match for the target intent.",
                "Answer no when another candidate is better or the query is ambiguous.",
                "Return exactly one word: yes or no.",
                f"Prompt version: {prompt_version}",
            ]
        else:
            lines = [
                "Instruction: Decide whether the query matches the target intent.",
                "Use the full candidate list before answering.",
                "Answer strictly with yes or no.",
                f"Prompt version: {prompt_version}",
            ]
        if candidate_intents:
            lines.append("Candidate intents:")
            separator = ":" if prompt_version in {"ranking_v2", "ranking_v3"} else "-"
            for idx, candidate in enumerate(candidate_intents, start=1):
                desc = LLMSemanticVerifier.intent_to_description(str(candidate))
                lines.append(f"{idx}. {candidate} {separator} {desc}")
        lines.extend(
            [
                f"Target intent: {intent_name}",
                f"Description: {LLMSemanticVerifier.intent_to_description(str(intent_name))}",
                f"Query: {query}",
                "Answer:",
            ]
        )
        return "\n".join(lines)

    def _build_prompt(
        self,
        query: str,
        intent_name: str,
        intent_desc: str,
        candidate_intents: Sequence[str] | None = None,
    ) -> str:
        """Build a candidate-aware decoder-only verification prompt."""
        prompt_version = str(self.config.prompt_version)
        if prompt_version == "ranking_v2":
            lines = [
                "You are a strict binary verifier for intent routing.",
                "Compare the query against the target intent and the competing candidates.",
                "Answer yes only when the target intent is the best semantic match.",
                "If the query is ambiguous or fits another candidate better, answer no.",
                "Return exactly one word: yes or no.",
                f"Prompt version: {prompt_version}",
            ]
        elif prompt_version == "ranking_v3":
            lines = [
                "You are a binary verifier for intent routing.",
                "Use the candidate list to compare the target intent against nearby alternatives.",
                "Answer yes when the query is a clear or strong match for the target intent.",
                "Answer no when another candidate is better or the query is ambiguous.",
                "Return exactly one word: yes or no.",
                f"Prompt version: {prompt_version}",
            ]
        else:
            lines = [
                "Instruction: Decide whether the query matches the target intent.",
                "Use the full candidate list before answering.",
                "Answer strictly with yes or no.",
                f"Prompt version: {prompt_version}",
            ]
        if candidate_intents:
            lines.append(self._build_candidate_block(candidate_intents, prompt_version))
        lines.extend(
            [
                f"Target intent: {intent_name}",
                f"Description: {intent_desc}",
                f"Query: {query}",
                "Answer:",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _score_from_logits(logits: torch.Tensor, yes_token_id: int, no_token_id: int) -> float:
        """Convert final-token logits into a normalized yes-probability score."""
        probs = torch.softmax(logits, dim=-1)
        yes_p = float(probs[0, yes_token_id].item())
        no_p = float(probs[0, no_token_id].item())
        denom = yes_p + no_p
        return yes_p / denom if denom > 1e-12 else 0.5

    @torch.no_grad()
    def verify(
        self,
        queries: Sequence[str],
        candidate_intents: Sequence[str],
        candidate_contexts: Sequence[Sequence[str]] | None = None,
    ) -> List[float]:
        """Return probability-like scores for semantic match (Yes probability)."""
        if len(queries) != len(candidate_intents):
            raise ValueError("queries and candidate_intents must have same length")
        if candidate_contexts is not None and len(candidate_contexts) != len(candidate_intents):
            raise ValueError("candidate_contexts must match candidate_intents length")

        scores: List[float] = []
        for idx, (query, intent_name) in enumerate(zip(queries, candidate_intents)):
            desc = self.intent_to_description(str(intent_name))
            candidate_context = None
            if candidate_contexts is not None:
                candidate_context = list(candidate_contexts[idx])
            prompt = self._build_prompt(
                str(query),
                str(intent_name),
                desc,
                candidate_intents=candidate_context,
            )

            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            base_model = getattr(self.model, "base", self.model)
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            scores.append(float(self._score_from_logits(logits, self.yes_token_id, self.no_token_id)))

        return scores

    @torch.no_grad()
    def score_candidate_intents(
        self,
        query_text: str,
        candidate_intents: Sequence[str],
        candidate_contexts: Sequence[Sequence[str]] | None = None,
    ) -> List[float]:
        """Score multiple candidate intents for one query using decoder-only likelihood."""
        return self.verify(
            [query_text] * len(candidate_intents),
            list(candidate_intents),
            candidate_contexts=candidate_contexts,
        )

    @torch.no_grad()
    def rank_candidate_intents(
        self,
        query_text: str,
        candidate_intents: Sequence[str],
    ) -> Dict[str, Any]:
        """Rank candidate intents by decoder-only semantic score."""
        candidate_list = [str(intent) for intent in candidate_intents]
        if not candidate_list:
            return {
                "candidate_intents": [],
                "candidate_scores": [],
                "best_intent": "",
                "best_score": 0.5,
                "runner_up_intent": "",
                "runner_up_score": 0.5,
                "score_margin": 0.0,
            }

        candidate_contexts = [candidate_list for _ in candidate_list]
        scores = self.score_candidate_intents(
            query_text,
            candidate_list,
            candidate_contexts=candidate_contexts,
        )
        ranked = sorted(zip(candidate_list, scores), key=lambda item: item[1], reverse=True)

        best_intent, best_score = ranked[0]
        runner_up_intent = ranked[1][0] if len(ranked) > 1 else ""
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.5
        return {
            "candidate_intents": [item[0] for item in ranked],
            "candidate_scores": [float(item[1]) for item in ranked],
            "best_intent": best_intent,
            "best_score": float(best_score),
            "runner_up_intent": runner_up_intent,
            "runner_up_score": float(runner_up_score),
            "score_margin": float(best_score - runner_up_score),
        }
