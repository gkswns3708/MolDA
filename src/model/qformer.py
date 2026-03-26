"""
Q-Former stub for Stage 1 (unused, placeholder for Stage 2+).
Actual implementation: SciBERT-based cross-attention with graph embeddings.
Reference: Old_MolDA/model/blip2.py (init_Qformer)
"""

import torch
import torch.nn as nn


class QFormer(nn.Module):
    """Stub. Will be implemented for Stage 2+."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, graph_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QFormer is not available in Stage 1.")
