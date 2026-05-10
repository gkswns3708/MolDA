"""RefMolDA: frozen reference policy for V-MolPO preference optimization.

Wraps a full MolDA instance loaded from a Stage 2 checkpoint (LoRA+embed+QFormer+GNN
weights) and frozen for the duration of preference training. Used by VRPOELBO to
compute B̂_ref(y) — the reference-policy ELBO term in the DPO-E score.

Why a separate full copy (vs LoRA gate toggle)?
  - PEFT의 disable_adapter() 는 base model 만 forward 함 → 그건 π_base, NOT π_ref
  - π_ref = base + Stage 2 LoRA weights (frozen at Stage 2 init).
    이를 표현하려면 (a) 별도 PEFT 인스턴스 또는 (b) multi-adapter PEFT.
    (b) 는 LoRA 가중치를 별도 adapter 로 저장/로드해야 해서 ckpt 형태 변경 필요.
    (a) 가 가장 단순하고 메모리도 RTX PRO 6000 (97GB) 에서 감당 가능.

Forward path:
  - String-only:        ref.llada.model(noisy_ids, attention_mask)
  - String+graph:       ref._forward_with_graph(noisy_ids, ...)  (uses ref's QFormer/GNN)

The provided forward_fn factory matches the signature expected by
src/training/vrpo_elbo.py compute_elbo: callable(noisy_ids, attention_mask) -> logits.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.model.molda import MolDA

logger = logging.getLogger(__name__)


def _extract_state_dict(ckpt: dict, strip_prefix: str = "model.") -> dict:
    """Pull state_dict from a Lightning ckpt and strip the trainer.model prefix.

    Lightning saves trainer.state_dict() as ckpt["state_dict"], where keys are
    prefixed with "model." (matching MolDATrainer.model attribute). RefMolDA
    loads directly into a MolDA instance, so the prefix must be stripped.

    If the ckpt has no "state_dict" key (raw weights dict), return as-is with
    prefix-stripping applied.
    """
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(strip_prefix):
            out[k[len(strip_prefix):]] = v
        else:
            out[k] = v
    return out


class RefMolDA(nn.Module):
    """Frozen reference π_ref. Loads Stage 2 ckpt and disables grad on all params.

    Use as a member on the trainable MolDA instance:
        self.ref_model = RefMolDA(cfg, ref_ckpt_path)
    """

    def __init__(self, cfg, ref_ckpt_path: str):
        super().__init__()
        self.cfg = cfg
        self.ref_ckpt_path = ref_ckpt_path

        # CRITICAL: disable molpo on inner MolDA to prevent infinite recursion.
        # MolDA.__init__ creates its own RefMolDA when cfg.molpo.enabled=True,
        # which would recursively load LLaDA-8B → OOM. Force-clone cfg with
        # molpo.enabled=False for the inner MolDA.
        from omegaconf import OmegaConf
        ref_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if "molpo" in ref_cfg:
            ref_cfg.molpo.enabled = False
        self.molda = MolDA(ref_cfg)

        self._load_ref_ckpt(ref_ckpt_path)
        self._freeze_all()

    @property
    def tokenizer(self):
        return self.molda.tokenizer

    def _load_ref_ckpt(self, path: str):
        """Load Stage 2 ckpt into self.molda. Mirrors CheckpointMixin.load_pretrained_state_dict
        but operates directly on the MolDA module (no Lightning wrapping).

        Lightning Trainer ckpts have keys prefixed with "model." (matching trainer.model = MolDA).
        Strip if present.
        """
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"RefMolDA ckpt not found: {path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        clean = _extract_state_dict(ckpt, strip_prefix="model.")
        # Drop ref_model.* keys (defense: re-using a Stage 3 ckpt as ref shouldn't
        # double-wrap)
        clean = {k: v for k, v in clean.items() if not k.startswith("ref_model.")}

        msg = self.molda.load_state_dict(clean, strict=False)
        logger.info(
            f"RefMolDA loaded from {path}: "
            f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
        )
        # Strict on unexpected: if ref ckpt has keys we don't recognize, the loaded
        # weights are likely from a different model architecture → silent ref policy
        # corruption. Better fail fast.
        if msg.unexpected_keys:
            raise RuntimeError(
                f"RefMolDA: unexpected keys in ref ckpt {path}. "
                f"This indicates ckpt/model architecture mismatch (silent ref policy "
                f"corruption risk). First 10: {msg.unexpected_keys[:10]}"
            )

    def _freeze_all(self):
        n = 0
        for p in self.molda.parameters():
            p.requires_grad = False
            n += p.numel()
        # Force self + all children to eval mode (recurses via nn.Module.eval())
        self.eval()
        logger.info(f"RefMolDA frozen: {n:,} params, all requires_grad=False")

    def train(self, mode: bool = True):
        """Always keep ref in eval mode regardless of trainer mode toggle."""
        return super().train(False)

    @torch.no_grad()
    def forward_logits(self, noisy_ids: torch.Tensor,
                       attention_mask: torch.Tensor | None = None,
                       batch: dict | None = None) -> torch.Tensor:
        """Forward without loss computation. Returns logits [B, L, V].

        If string+graph mode and `batch` provided with `graphs`, uses MolDA's
        graph injection path (Q-Former output → <mol> token positions).
        """
        use_graph = (
            self.molda.stage >= 2
            and self.molda.mol_representation == "string+graph"
            and batch is not None
            and "graphs" in batch
        )
        if use_graph:
            outputs = self.molda._forward_with_graph(
                noisy_ids, attention_mask, batch, batch["input_ids"]
            )
        else:
            outputs = self.molda.llada.model(
                input_ids=noisy_ids, attention_mask=attention_mask
            )
        return outputs.logits

    def make_forward_fn(self, batch: dict | None = None):
        """Return a callable matching compute_elbo's forward_fn signature.

        Bind the batch (for graph injection) once per training step.
        """
        @torch.no_grad()
        def fwd(noisy_ids, attention_mask=None):
            return self.forward_logits(noisy_ids, attention_mask, batch=batch)
        return fwd
