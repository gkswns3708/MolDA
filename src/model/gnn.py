"""GNN adapter: wraps the vendored GINE_TokenGT (src/model/adapter/gine_tokengt.py).

Translates the new MolDA cfg.gnn schema into the args-namespace shape the
vendored Old_MolDA GINE_TokenGT expects. Forward signature matches the
original (x, edge_index, edge_attr, batch) -> (mol_embeds, mol_masks).
"""

from types import SimpleNamespace

import torch.nn as nn

from src.model.adapter.gine_tokengt import GINE_TokenGT as _VendoredGINETokenGT


def _build_old_args(cfg):
    """Translate cfg.gnn (new schema) -> SimpleNamespace (old schema)."""
    return SimpleNamespace(
        gine=SimpleNamespace(
            gin_num_layers=cfg.gnn.gine.num_layers,
            gnn_hidden_dim=cfg.gnn.gine.hidden_dim,
            drop_ratio=cfg.gnn.gine.drop_ratio,
            gnn_jk=cfg.gnn.gine.jk,
            graph_encoder_ckpt=cfg.gnn.gine.ckpt,
        ),
        tokengt=SimpleNamespace(
            input_feat_dim=cfg.gnn.tokengt.input_feat_dim,
            gnn_hidden_dim=cfg.gnn.tokengt.hidden_dim,
            num_layers=cfg.gnn.tokengt.num_layers,
            num_heads=cfg.gnn.tokengt.num_heads,
            method=cfg.gnn.tokengt.method,
            d_p=cfg.gnn.tokengt.d_p,
            d_e=cfg.gnn.tokengt.d_e,
            use_graph_token=cfg.gnn.tokengt.use_graph_token,
            max_position_embeddings=cfg.gnn.tokengt.max_position_embeddings,
            graph_encoder_ckpt=cfg.gnn.tokengt.ckpt,
        ),
        debug=cfg.get("debug", False),
    )


class GINETokenGT(nn.Module):
    """GINE + TokenGT parallel encoder (concat along sequence dim)."""

    def __init__(self, cfg):
        super().__init__()
        self.tune_gnn = bool(cfg.model.get("tune_gnn", False))
        self.encoder = _VendoredGINETokenGT(_build_old_args(cfg))

        if not self.tune_gnn:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        """Override so that frozen pretrained encoder stays in eval mode even when
        Lightning calls model.train(). Mirrors Old_MolDA blip2.py's disabled_train,
        but pickle-safe (no instance-level method patching).
        """
        super().train(mode)
        if not self.tune_gnn:
            self.encoder.eval()
        return self

    def forward(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)
