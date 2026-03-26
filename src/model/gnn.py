"""
GNN stub for Stage 1 (unused, placeholder for Stage 2+).
Actual implementation: GINE + TokenGT parallel encoding → concat.
Reference: Old_MolDA/model/gine_tokengt.py
"""

import torch
import torch.nn as nn


class GINETokenGT(nn.Module):
    """Stub. Will be implemented for Stage 2+."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, graph_batch) -> torch.Tensor:
        raise NotImplementedError("GINETokenGT is not available in Stage 1.")
