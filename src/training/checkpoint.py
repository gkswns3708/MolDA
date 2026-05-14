"""CheckpointMixin: checkpoint 저장 시 frozen 파라미터 필터링.

trainable params (LoRA, embed, head, PEFT, GNN/QFormer)만 보존하여
체크포인트 크기를 줄임.
"""

import logging

import torch

from src.training.optimizer import _is_output_head_param

logger = logging.getLogger(__name__)


class CheckpointMixin:
    """Checkpoint 저장 시 trainable params만 보존하는 Mixin."""

    # Checkpoints only contain trainable params (frozen base weights stripped in
    # on_save_checkpoint). On load, default Lightning strict=True would fail because
    # the base-model keys are present in the live module but absent from the ckpt.
    # The base weights come from the HF model load during model init, so dropping
    # strict is safe for test/validate/resume flows alike.
    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        return super().load_state_dict(state_dict, strict=False, *args, **kwargs)

    def load_pretrained_state_dict(self, path: str):
        """Stage N→N+1 weight 이전. 가중치만 로드 (optimizer/scheduler/epoch 미복원).

        Stage 1 ckpt는 trainable params만 담고 있으므로(on_save_checkpoint 필터),
        missing_keys = base LLM weight 정도가 정상 — 이미 HF init에서 로드됨.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Loaded pretrained from {path}: "
            f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
        )
        if msg.unexpected_keys:
            logger.warning(f"  Unexpected keys (first 5): {msg.unexpected_keys[:5]}")

    def on_save_checkpoint(self, checkpoint):
        to_remove = []
        for key in checkpoint["state_dict"]:
            # Drop ref_model.* (frozen reference policy, always reloadable from ref_ckpt_path)
            if key.startswith("model.ref_model."):
                to_remove.append(key)
                continue
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

        # V-MolPO per-task EMA anchor: persist outside state_dict
        ema = getattr(self.model, "task_anchor_ema", None)
        if ema is not None:
            checkpoint["v_molpo_task_anchor_ema"] = ema.state_dict()

        logger.info(f"Checkpoint: kept {len(checkpoint['state_dict'])} params, "
                    f"removed {len(to_remove)} frozen/ref params"
                    + (f", + EMA({len(ema)} tasks)" if ema is not None else ""))

    def on_load_checkpoint(self, checkpoint):
        # V-MolPO per-task EMA anchor: restore if present (epoch-cross persistence)
        ema = getattr(self.model, "task_anchor_ema", None)
        if ema is not None and "v_molpo_task_anchor_ema" in checkpoint:
            ema.load_state_dict(checkpoint["v_molpo_task_anchor_ema"])
            logger.info(
                f"Restored V-MolPO task_anchor_ema with {len(ema)} task entries"
            )
