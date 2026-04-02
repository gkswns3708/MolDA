"""
LLaDA model wrapper: tokenizer + model loading + LoRA + vocab expansion.

Responsibilities:
1. Load tokenizer + add special tokens (added_tokens.py + optional mol dict)
2. Load LLaDA model (AutoModelForCausalLM) with weight_tying=True override
3. Resize embeddings with mean + std*randn initialization for new tokens
4. Apply LoRA via PEFT
5. Set trainability: LoRA + wte(tied to output) trainable, rest frozen

Reference: Old_MolDA/model/blip2_llada.py (check_and_add_special_tokens, set_llm_model)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.model import added_tokens

logger = logging.getLogger(__name__)


class LLaDAWrapper(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._original_vocab_size = cfg.model.original_vocab_size

        # 1. Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.llm, trust_remote_code=True
        )

        # 2. Add special tokens
        n_added = self._add_special_tokens()
        logger.info(f"Added {n_added} special tokens. Vocab: {len(self._tokenizer)}")

        # 3. Load model with weight_tying override
        #    - weight_tying=True: wte가 input+output 모두 담당 (ff_out 버려짐)
        #    - weight_tying=False: wte(input)와 ff_out(output) 별도 사용
        self._weight_tying = getattr(cfg.model, "weight_tying", True)
        llm_config = AutoConfig.from_pretrained(
            cfg.model.llm, trust_remote_code=True
        )
        llm_config.weight_tying = self._weight_tying
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model.llm,
            config=llm_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 4. Resize embeddings with mean init
        self._resize_embeddings_mean()

        # 5. Apply LoRA
        self._apply_lora()

        # 6. Log trainable parameters
        if self._weight_tying:
            logger.info("weight_tying=True: wte managed by PEFT modules_to_save (tied to output)")
        else:
            logger.info("weight_tying=False: wte + ff_out managed by PEFT modules_to_save (separate)")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def _add_special_tokens(self) -> int:
        """Add special tokens + mol repr tag + optional mol dict to tokenizer."""
        mol_token_type = self.cfg.tokenizer.mol_token_type  # "selfies" | "smiles"

        # Base special tokens (always added)
        special_tokens = (
            added_tokens.BOOL
            + added_tokens.FLOAT
            + added_tokens.DESCRIPTION
            + added_tokens.MOL_2D
            + added_tokens.MOL_3D
            + added_tokens.MOL_EMBEDDING
            + added_tokens.NUMBER
            + added_tokens.INSTRUCTION
            + added_tokens.REACTION_DIRECTION
            + added_tokens.IUPAC
            + added_tokens.MOLFORMULA
        )

        # Mol representation tag (always, one of)
        if mol_token_type == "selfies":
            special_tokens += added_tokens.SELFIES
        elif mol_token_type == "smiles":
            special_tokens += added_tokens.SMILES
        else:
            raise ValueError(f"Unknown mol_token_type: {mol_token_type}")

        # Mol dictionary (optional)
        if self.cfg.tokenizer.add_mol_dict:
            if mol_token_type == "selfies":
                selfies_path = Path(self.cfg.tokenizer.selfies_dict_path)
                if selfies_path.exists():
                    with open(selfies_path) as f:
                        selfies_tokens = [line.strip() for line in f if line.strip()]
                    special_tokens.extend(selfies_tokens)
                    logger.info(f"Loaded {len(selfies_tokens)} SELFIES tokens from {selfies_path}")
                else:
                    logger.warning(f"SELFIES token file not found: {selfies_path}")
            elif mol_token_type == "smiles":
                raise NotImplementedError(
                    "SMILES dictionary is not available yet. "
                    "Set tokenizer.add_mol_dict=false to use SMILES without dictionary."
                )

        n_added = self._tokenizer.add_tokens(special_tokens)

        # Store mol token id for later use
        self._tokenizer.mol_token_id = self._tokenizer.convert_tokens_to_ids("<mol>")

        return n_added

    def _resize_embeddings_mean(self):
        """Resize input + output embeddings, init new tokens with mean+std*randn."""
        new_vocab_size = len(self._tokenizer)

        # --- Input embeddings (wte) ---
        old_input_emb = self._model.get_input_embeddings()
        old_weight = old_input_emb.weight.data
        old_num = old_weight.shape[0]
        old_mean = old_weight.mean(dim=0)
        old_std = old_weight.std(dim=0)

        self._model.resize_token_embeddings(new_vocab_size)

        if new_vocab_size > old_num:
            new_input_emb = self._model.get_input_embeddings()
            num_new = new_vocab_size - old_num
            with torch.no_grad():
                new_input_emb.weight.data[-num_new:] = (
                    old_mean + old_std * torch.randn(
                        num_new, old_weight.shape[1],
                        device=old_weight.device, dtype=old_weight.dtype
                    )
                )
            logger.info(f"Input embed resized: {old_num} → {new_vocab_size} "
                        f"(+{num_new} mean-init tokens)")

        # --- Output embeddings (ff_out / lm_head) ---
        output_emb = self._model.get_output_embeddings()
        if output_emb is not None and output_emb.weight.shape[0] != new_vocab_size:
            old_out_weight = output_emb.weight.data
            n_orig = old_out_weight.shape[0]
            out_mean = old_out_weight.mean(dim=0)
            out_std = old_out_weight.std(dim=0)

            new_lm_head = nn.Linear(
                output_emb.in_features,
                new_vocab_size,
                bias=output_emb.bias is not None,
            ).to(device=old_out_weight.device, dtype=old_out_weight.dtype)

            num_new_out = new_vocab_size - n_orig
            with torch.no_grad():
                new_lm_head.weight[:n_orig, :] = old_out_weight
                if output_emb.bias is not None:
                    new_lm_head.bias[:n_orig] = output_emb.bias
                if num_new_out > 0:
                    new_lm_head.weight[n_orig:, :] = (
                        out_mean + out_std * torch.randn(
                            num_new_out, old_out_weight.shape[1],
                            device=old_out_weight.device, dtype=old_out_weight.dtype
                        )
                    )

            self._model.set_output_embeddings(new_lm_head)
            logger.info(f"Output embed resized: {n_orig} → {new_vocab_size} "
                        f"(+{num_new_out} mean-init tokens)")

    def _apply_lora(self):
        """Apply LoRA to attention and MLP layers."""
        lora_cfg = self.cfg.lora
        lora_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            modules_to_save=list(lora_cfg.modules_to_save),
            bias="none",
        )
        self._model = get_peft_model(self._model, lora_config)
        logger.info(f"LoRA applied: r={lora_cfg.r}, alpha={lora_cfg.alpha}")
        self._model.print_trainable_parameters()


