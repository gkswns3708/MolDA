"""
Shared fixtures and markers for MolDA test suite.

Uses OmegaConf (not Hydra) to avoid GlobalHydra state conflicts in pytest.
"""

import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")

# ── HuggingFace cache ──
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "hf-cache"))


# ── Pytest markers ──

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU (CUDA)")
    config.addinivalue_line("markers", "slow: slow tests (model loading, >10s)")
    config.addinivalue_line("markers", "integration: integration tests spanning multiple components")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when no CUDA available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ── Config fixture (OmegaConf, NOT Hydra) ──

@pytest.fixture(scope="session")
def cfg():
    """Minimal config matching toy.yaml + default.yaml + stage1.yaml."""
    return OmegaConf.create({
        "seed": 42,
        "mode": "ft",
        "stage": 1,
        "debug": False,
        "ckpt_path": None,
        "model": {
            "llm": "GSAI-ML/LLaDA-8B-Instruct",
            "tune_llm": "lora",
            "tune_gnn": False,
            "mol_representation": "string_only",
            "original_vocab_size": 126349,
        },
        "lora": {
            "r": 64,
            "alpha": 32,
            "dropout": 0.05,
            "config_path": None,
        },
        "tokenizer": {
            "add_selfies_tokens": True,
            "selfies_token_path": os.path.join(PROJECT_ROOT, "src", "model", "selfies_dict.txt"),
        },
        "data": {
            "root": DATASET_ROOT,
            "splits": {
                "train": "Train_toy100",
                "val": "Val_toy100",
                "test": "Test_toy100",
            },
            "max_length": 512,
            "gen_max_len": 256,
            "truncation": True,
            "padding": "max_length",
            "min_len": 8,
        },
        "training": {
            "max_steps": -1,
            "max_epochs": 3,
            "batch_size": 4,
            "accumulate_grad_batches": 1,
            "weight_decay": 0.1,
            "gradient_clip_val": 1.0,
        },
        "scheduler": {
            "warmup_steps": 50,
            "decay_ratio": 0.1,
            "min_lr_ratio": 0.1,
        },
        "lr": {
            "lora": 2.5e-3,
            "embed_orig": 2.5e-5,
            "embed_new": 2.5e-5,
            "head_orig": 2.5e-5,
            "head_new": 2.5e-5,
            "other": 0.0,
        },
        "hardware": {
            "accelerator": "gpu",
            "devices": "0",
            "precision": "bf16-mixed",
            "num_workers": 0,
            "find_unused_parameters": True,
        },
        "generation": {
            "remasking_strategy": "low_confidence",
            "sampling_steps": 32,
            "semi_ar": {
                "enabled": False,
                "block_size": 32,
                "steps_per_block": 4,
            },
        },
        "validation": {
            "num_sanity_val_steps": 0,
            "val_check_interval": 1.0,
            "check_val_every_n_epoch": 1,
            "limit_val_batches": 1.0,
            "inference_batch_size": 8,
        },
        "logging": {
            "dir": "/tmp/molda_test_ckpt",
            "log_every_n_steps": 1,
            "save_on_n_steps": 500,
            "save_top_k_checkpoints": -1,
            "save_every_n_epochs": 1,
            "val_log_samples_per_gpu": 1,
            "log_stepwise_denoising": False,
            "stepwise_max_samples": 8,
            "log_nan_details": True,
            "nan_log_dir": "/tmp/molda_test_nan",
        },
        "wandb": {"enabled": False},
        "qformer": {
            "num_query_token": 32,
            "bert_name": "scibert",
            "bert_hidden_dim": 768,
            "cross_attention_freq": 2,
            "num_layers": 5,
        },
    })


# ── Real tokenizer (downloads tokenizer only, not model weights) ──

@pytest.fixture(scope="session")
def real_tokenizer():
    """Load actual LLaDA tokenizer with special tokens added."""
    from transformers import AutoTokenizer
    from src.model import added_tokens

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
    )

    special_tokens = (
        added_tokens.BOOL + added_tokens.FLOAT + added_tokens.DESCRIPTION
        + added_tokens.SELFIES + added_tokens.MOL_2D + added_tokens.MOL_3D
        + added_tokens.MOL_EMBEDDING + added_tokens.NUMBER
        + added_tokens.INSTRUCTION + added_tokens.REACTION_DIRECTION
        + added_tokens.IUPAC + added_tokens.MOLFORMULA
    )
    tokenizer.add_tokens(special_tokens)
    return tokenizer


# ── Toy dataset fixtures ──

@pytest.fixture(scope="session")
def toy_train_dataset():
    """Load Train_toy100 dataset."""
    from src.data.dataset import MoleculeDataset
    return MoleculeDataset(os.path.join(DATASET_ROOT, "Train_toy100"))


@pytest.fixture(scope="session")
def toy_train_samples(toy_train_dataset):
    """Load first 10 samples from Train_toy100."""
    return [toy_train_dataset[i] for i in range(min(10, len(toy_train_dataset)))]


# ── GPU model fixtures (session-scoped, 1개의 모델만 GPU에 로드) ──

@pytest.fixture(scope="session")
def trainer_module(cfg):
    """Load MolDATrainer ONCE for the entire test session.

    학습 시에는 모델 1개만 GPU에 올라가듯,
    테스트에서도 session-scoped로 1개만 올려 OOM을 방지한다.

    MolDATrainer 내부에 MolDA가 있고, MolDA 내부에 LLaDAWrapper가 있으므로
    이 fixture 하나로 모든 GPU 테스트가 모델을 공유한다.
    """
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    from src.training.trainer import MolDATrainer
    trainer = MolDATrainer(cfg)
    trainer = trainer.cuda()
    return trainer


@pytest.fixture(scope="session")
def molda_model(trainer_module):
    """MolDA model reference from the shared MolDATrainer.

    trainer_module.model이므로 별도 GPU 메모리를 사용하지 않는다.
    """
    return trainer_module.model


@pytest.fixture(scope="session")
def llada_wrapper(molda_model):
    """LLaDAWrapper reference from the shared MolDA model."""
    return molda_model.llada
