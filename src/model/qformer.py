"""QFormer adapter: BLIP-2 style Q-Former + ln_graph + opt_proj.

Reproduces the Stage 2 bridge from Old_MolDA (Blip2_LLaDA) using the same
init_Qformer helper from the vendored Blip2Base. Exposes Qformer / query_tokens
/ ln_graph / opt_proj as submodules so the freeze policy in molda.py
(STAGE2_TRAINABLE_KEYS) keeps matching by substring.
"""

import torch.nn as nn

from src.model.adapter.blip2 import Blip2Base, LayerNorm


class QFormer(nn.Module):
    """BLIP-2 Q-Former bridge + LN + projection to LLM hidden dim."""

    def __init__(self, cfg, llm_hidden_dim):
        super().__init__()

        qcfg = cfg.qformer
        bert_name = qcfg.get("bert_name", "scibert")
        num_query_token = int(qcfg.num_query_token)
        cross_attention_freq = int(qcfg.get("cross_attention_freq", 2))
        bert_num_hidden_layers = int(
            qcfg.get("num_layers", qcfg.get("bert_num_hidden_layers", -1))
        )
        # GINE+TokenGT concat along sequence dim → feature dim stays at gine.hidden_dim.
        graph_width = int(cfg.gnn.gine.hidden_dim)

        self.Qformer, self.query_tokens = Blip2Base.init_Qformer(
            bert_name,
            num_query_token,
            graph_width,
            cross_attention_freq=cross_attention_freq,
            bert_num_hidden_layers=bert_num_hidden_layers,
        )
        # Mirror Blip2Qformer.__init__: resize embeddings for [DEC] bos token + copy
        # base attention weights into the cross-attention "_query" copies.
        scibert_tokenizer = Blip2Base.init_tokenizer()
        self.Qformer.resize_token_embeddings(len(scibert_tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                if key_orig in state_dict:
                    param.data.copy_(state_dict[key_orig])

        # Drop modules that are never invoked on the query-embed-only forward
        # path. Keeping them would create unused params that break DDP under
        # find_unused_parameters=False (BLIP-2 standard cleanup).
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.ln_graph = LayerNorm(graph_width)
        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, llm_hidden_dim)

    def forward(self, mol_embeds, mol_masks):
        """mol_embeds: [B,S,graph_width], mol_masks: [B,S] -> [B,num_query,llm_hidden]."""
        mol_embeds = self.ln_graph(mol_embeds, mol_masks)
        query_tokens = self.query_tokens.expand(mol_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=mol_embeds,
            encoder_attention_mask=mol_masks,
            return_dict=True,
        )
        return self.opt_proj(query_output.last_hidden_state)
