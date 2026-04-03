"""CheckpointMixin: checkpoint 저장 시 frozen 파라미터 필터링.

trainable params (LoRA, embed, head, PEFT, GNN/QFormer)만 보존하여
체크포인트 크기를 줄임.
"""

import logging

from src.training.optimizer import _is_output_head_param

logger = logging.getLogger(__name__)


class CheckpointMixin:
    """Checkpoint 저장 시 trainable params만 보존하는 Mixin."""

    def on_save_checkpoint(self, checkpoint):
        to_remove = []
        for key in checkpoint["state_dict"]:
            # Keep: lora params, embedding, output head, PEFT wrappers, GNN/QFormer
            keep = (
                any(k in key for k in [
                    "lora_", "embed", "wte", "lm_head", "output_embeddings",
                    "modules_to_save", "original_module",
                    "qformer", "gnn", "query_tokens", "opt_proj", "ln_graph",
                ])
                or _is_output_head_param(key)
            )
            if not keep:
                to_remove.append(key)

        for key in to_remove:
            del checkpoint["state_dict"][key]

        logger.info(f"Checkpoint: kept {len(checkpoint['state_dict'])} params, "
                    f"removed {len(to_remove)} frozen params")
