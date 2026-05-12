import torch
import torch.nn as nn

# from transformers import BertTokenizer
from src.model.adapter.gin_model import GNN, GNN_MoleculeSTM
from src.model.adapter.tokenGT import BERTTokenGT


def _strip_prefix(state_dict, candidates):
    """Try each prefix in `candidates`; rename matching keys (drop prefix).

    Returns a fresh dict containing only keys that matched the first hit prefix.
    Raises KeyError if no candidate prefix matches any key.
    """
    for prefix in candidates:
        if any(k.startswith(prefix) for k in state_dict):
            return {
                k.replace(prefix, "", 1): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
    raise KeyError(
        f"None of the candidate prefixes {candidates} matched any key in state_dict. "
        f"Sample keys: {list(state_dict.keys())[:5]}"
    )


class GINE_TokenGT(nn.Module):
    def __init__(self, args):
        super(GINE_TokenGT, self).__init__()
        self.graph_encoder_gine = GNN_MoleculeSTM(
            num_layer=args.gine.gin_num_layers,
            emb_dim=args.gine.gnn_hidden_dim,
            gnn_type="gin",
            drop_ratio=args.gine.drop_ratio,
            JK=args.gine.gnn_jk,
            args=args,
        )
        self.graph_encoder_tokengt = BERTTokenGT(
            input_feat_dim=args.tokengt.input_feat_dim,
            hidden_dim=args.tokengt.gnn_hidden_dim,
            num_layers=args.tokengt.num_layers,
            num_heads=args.tokengt.num_heads,
            method=args.tokengt.method,
            d_p=args.tokengt.d_p,
            d_e=args.tokengt.d_e,
            use_graph_token=args.tokengt.use_graph_token,
            max_position_embeddings=args.tokengt.max_position_embeddings
        )
        ##### load pretrained GINE #####
        # Pretrained ckpt prefixes vary:
        #   - raw MoleculeSTM/TokenGT (Custom_gnn_models): keys like `gnn.X`
        #   - Stage 2 wrapped Blip2_LLaDA ckpts: keys like
        #     `blip2model.graph_encoder.graph_encoder_{gine,tokengt}.X`
        # Try both prefixes (rename the first matching) and load with strict=True.
        if getattr(args, 'debug', False):
            print(args.gine.graph_encoder_ckpt, "-args.gine.graph_encoder_ckpt")
        ckpt = torch.load(
            args.gine.graph_encoder_ckpt, map_location=torch.device("cpu"), weights_only=False
        )
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        renamed_state_dict = _strip_prefix(
            sd,
            candidates=("blip2model.graph_encoder.graph_encoder_gine.", "gnn."),
        )
        self.graph_encoder_gine.load_state_dict(renamed_state_dict, strict=True)
        if getattr(args, 'debug', False):
            print(f"load graph encoder from {args.gine.graph_encoder_ckpt}")

        ##### load pretrained TokenGT #####
        ckpt = torch.load(
            args.tokengt.graph_encoder_ckpt, map_location=torch.device("cpu"), weights_only=False
        )
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        renamed_state_dict = _strip_prefix(
            sd,
            candidates=("blip2model.graph_encoder.graph_encoder_tokengt.", "gnn."),
        )
        self.graph_encoder_tokengt.load_state_dict(renamed_state_dict, strict=True)
        if getattr(args, 'debug', False):
            print(f"load graph encoder from {args.tokengt.graph_encoder_ckpt}")

        self.layer_norm_gine = nn.LayerNorm(args.gine.gnn_hidden_dim)
        self.layer_norm_tokengt = nn.LayerNorm(args.tokengt.gnn_hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        gine_output, gine_mask = self.graph_encoder_gine(x, edge_index, edge_attr, batch)
        tokengt_output, tokengt_mask = self.graph_encoder_tokengt(x, edge_index, edge_attr, batch)

        # apply layer normalization
        gine_output = self.layer_norm_gine(gine_output)
        tokengt_output = self.layer_norm_tokengt(tokengt_output)

        # concatenate the outputs
        output = torch.concat((gine_output, tokengt_output), dim=1)
        mask = torch.concat((gine_mask, tokengt_mask), dim=1)

        return output, mask
