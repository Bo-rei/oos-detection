"""ExpertManager: Memory-efficient Expert inference with shared base + adapter switching"""
import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.models.expert import SmolLMExpert


class ExpertManager:
    """Manages a single shared SmolLM-135M base with 10 LoRA adapters + projection heads.
    
    Memory-efficient design:
    - Load one SmolLM-135M base (576 hidden dim)
    - Register 10 LoRA adapters (one per domain)
    - Store 10 classification heads + 10 intent centroids
    - Switch adapter/head dynamically per inference call
    """

    def __init__(self, base_model_path: str, device: str = 'cuda'):
        """Initialize with a shared base model.
        
        Args:
            base_model_path: Path to SmolLM-135M base model
            device: Device to load model on
        """
        self.device = device
        self.base_model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='right', trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Shared base (initially without any adapter)
        self.base_model = SmolLMExpert(base_model_path, projection_dim=128, lora_r=8, lora_alpha=32)
        self.base_model = self.base_model.to(device)
        self.base_model.eval()
        
        # Domain registry
        self.domains: Dict[str, Dict] = {}  # domain -> {adapter_state, cls_head, centroids, intent_names}
        self.current_domain: Optional[str] = None

    def register_adapter(self, domain: str, adapter_dir: str):
        """Register a domain's adapter, classification head, and centroids.
        
        Args:
            domain: Domain name (e.g., 'banking')
            adapter_dir: Path to directory containing best_model.pt and intent_centroids.npy
        """
        ckpt_path = os.path.join(adapter_dir, 'best_model.pt')
        centroids_path = os.path.join(adapter_dir, 'intent_centroids.npy')
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        if not os.path.exists(centroids_path):
            raise FileNotFoundError(f'Centroids not found: {centroids_path}')
        
        # Some checkpoints contain numpy objects that require legacy unpickling.
        # Try the safe default first and fall back to weights_only=False when needed.
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        except Exception:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        adapter_state = ckpt.get('adapter')
        head_state = ckpt.get('head')
        projection_state = ckpt.get('projection')
        if projection_state is None:
            logging.warning(
                'Domain %s: checkpoint has no "projection" key. '
                'Projection head will NOT be swapped on activate(); '
                'multi-domain inference will be incorrect.', domain
            )
        centroids = np.load(centroids_path)
        
        # Reconstruct classification head
        num_intents = centroids.shape[0]
        cls_head = nn.Linear(128, num_intents).to(self.device)
        cls_head.load_state_dict(head_state)
        cls_head.eval()
        
        self.domains[domain] = {
            'adapter_state': adapter_state,
            'projection_state': projection_state,
            'cls_head': cls_head,
            'centroids': torch.tensor(centroids, dtype=torch.float32, device=self.device),
            'num_intents': num_intents
        }

    def activate(self, domain: str):
        """Switch to a specific domain's adapter.
        
        Args:
            domain: Domain name to activate
        """
        if domain not in self.domains:
            raise ValueError(f'Domain {domain} not registered. Available: {list(self.domains.keys())}')
        
        if self.current_domain == domain:
            return  # already active
        
        # Load adapter state into base model
        adapter_state = self.domains[domain]['adapter_state']
        # PEFT models store adapter weights under 'base_model.model.xxx.lora_A.weight' etc.
        # We need to load them correctly into self.base_model.base (which is a PeftModel)
        try:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.base_model.base, adapter_state)
        except Exception as e:
            logging.warning(f'PEFT set_state_dict failed ({e}), trying strict=False load')
            self.base_model.base.load_state_dict(adapter_state, strict=False)

        # Load projection head (domain-specific; must be swapped together with the adapter)
        projection_state = self.domains[domain].get('projection_state')
        if projection_state is not None:
            self.base_model.projection.load_state_dict(projection_state)
            logging.debug('Loaded projection head for domain: %s', domain)
        else:
            logging.warning(
                'Domain %s: projection state unavailable; '
                'retrained experts needed for correct multi-domain inference.', domain
            )

        self.current_domain = domain
        logging.debug(f'Activated domain: {domain}')

    def predict(self, domain: str, text: str, max_length: int = 128, return_features: bool = False, return_energy: bool = False, temperature: float = 1.0) -> Tuple[int, float, Optional[torch.Tensor]]:
        """Predict intent and confidence for a given text in a specific domain.
        
        Backwards-compatible: default behaviour is unchanged. New option `return_energy`
        computes the Energy Score used for OOD detection:
          E(x) = -T * logsumexp(logits / T)
        Higher E(x) -> more likely OOD.

        Args:
            domain: Domain to use for prediction
            text: Input text
            max_length: Max tokenization length
            return_features: If True, return normalized feature vector
            return_energy: If True, also return energy score (scalar)
            temperature: Temperature T used in energy calculation
        
        Returns:
            (predicted_intent_id, confidence, features) or
            (predicted_intent_id, confidence, features, energy) if return_energy
        """
        self.activate(domain)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Extract features (L2-normalized 128-d)
        with torch.no_grad():
            features = self.base_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )  # [1, 128]
        
        # Classification via classifier head (for speed) or centroid distance
        cls_head = self.domains[domain]['cls_head']
        centroids = self.domains[domain]['centroids']
        
        with torch.no_grad():
            logits = cls_head(features)  # [1, num_intents]
            pred_id = logits.argmax(dim=-1).item()
            
            # Confidence = cosine similarity to predicted centroid
            # (features and centroids are L2-normalized, so cosine = dot product)
            confidence = (features @ centroids[pred_id]).item()  # scalar
            # Clamp to [0, 1] for safety
            confidence = max(0.0, min(1.0, confidence))

            # Energy score (higher => more likely OOD). Keep T as a tunable param.
            if return_energy:
                # logits: [1, C]
                # Energy (NeurIPS 2020): E(x) = -T * logsumexp(logits / T)
                # Larger E(x) (closer to zero) => more likely OOD. We return that scalar.
                energy = - float((temperature * torch.logsumexp(logits / temperature, dim=-1)).squeeze().cpu().numpy())
                # energy is a float (typically negative); higher (less negative) => more OOD
            else:
                energy = None
        
        feat_out = features.squeeze(0) if return_features else None
        if return_energy:
            return pred_id, confidence, feat_out, energy
        return pred_id, confidence, feat_out

    def predict_with_all_distances(self, domain: str, text: str, max_length: int = 128) -> Dict[int, float]:
        """Compute cosine similarity to ALL centroids (for calibration/threshold search).
        
        Args:
            domain: Domain to use
            text: Input text
            max_length: Max tokenization length
        
        Returns:
            dict mapping intent_id -> cosine_similarity
        """
        self.activate(domain)
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            features = self.base_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )  # [1, 128]
            centroids = self.domains[domain]['centroids']
            similarities = (features @ centroids.T).squeeze(0)  # [num_intents]
        
        return {i: sim.item() for i, sim in enumerate(similarities)}
