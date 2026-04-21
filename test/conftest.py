"""
Shared fixtures and markers for MolDA test suite.

Uses OmegaConf (not Hydra) to avoid GlobalHydra state conflicts in pytest.
Config is loaded from actual yaml files, mimicking Hydra composition.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_DIR = Path(PROJECT_ROOT) / "src" / "configs"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")

# ── HuggingFace cache ──
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "hf-cache"))


# ── NLTK resources (caption metrics need punkt_tab + wordnet) ──
# Downloaded once per test session to avoid LookupError in TestCaptionEvaluate.
def _ensure_nltk_resources():
    import nltk
    required = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, download_name in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


_ensure_nltk_resources()


# ── Pytest markers ──

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU (CUDA)")
    config.addinivalue_line("markers", "slow: slow tests (model loading, >10s)")
    config.addinivalue_line("markers", "integration: integration tests spanning multiple components")
    config.addinivalue_line("markers", "dataset: tests requiring the committed dataset/Processed/toy100 fixture")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when no CUDA available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ── Config loading (OmegaConf yaml merge, NOT Hydra) ──

def load_config(config_name: str = "toy_SELFIES") -> OmegaConf:
    """Load config by merging yaml files, mimicking Hydra composition.

    Merge order:
        default.yaml → stage.yaml → data.yaml → top-level.yaml → test overrides

    Args:
        config_name: Top-level config name (e.g. "toy_SELFIES", "toy_SMILES")
    """
    # 1. Parse top-level config to find trainer/data defaults
    top_raw = OmegaConf.load(CONFIG_DIR / f"{config_name}.yaml")
    trainer_name = "stage1"
    data_name = None
    for item in OmegaConf.to_container(top_raw.get("defaults", [])):
        if isinstance(item, dict):
            if "trainer" in item:
                trainer_name = item["trainer"]
            elif "data" in item:
                data_name = item["data"]

    # 2. Trainer configs: default → stage override
    base = OmegaConf.load(CONFIG_DIR / "trainer" / "default.yaml")
    stage = OmegaConf.load(CONFIG_DIR / "trainer" / f"{trainer_name}.yaml")
    if "defaults" in stage:
        OmegaConf.update(stage, "defaults", None)
        del stage["defaults"]

    # 3. Data config (keys go under data.*)
    data_overlay = {}
    if data_name:
        data_overlay = {"data": OmegaConf.load(CONFIG_DIR / "data" / f"{data_name}.yaml")}

    # 4. Top-level overrides (strip Hydra-only keys)
    top = OmegaConf.to_container(top_raw, resolve=False)
    for key in ("defaults", "hydra"):
        top.pop(key, None)

    # 5. Merge all layers
    merged = OmegaConf.merge(base, stage, data_overlay, top)

    # 6. Test-specific overrides
    test_overrides = {
        "hardware": {"devices": "0"},
        "training": {"global_batch_size": 4},  # 1 device × batch_size 4 × accum 1
        "data": {"root": os.path.join(DATASET_ROOT, "Processed", "toy100")},
        "tokenizer": {
            "selfies_dict_path": os.path.join(PROJECT_ROOT, "src", "model", "selfies_dict.txt"),
        },
        "logging": {
            "dir": "/tmp/molda_test_ckpt",
            "nan_log_dir": "/tmp/molda_test_nan",
        },
    }
    merged = OmegaConf.merge(merged, test_overrides)

    return merged


# ── Config fixture ──

@pytest.fixture(scope="session")
def cfg():
    """Load config from yaml files (toy_SELFIES + stage1 + default)."""
    return load_config("toy_SELFIES")


# ── Real tokenizer (downloads tokenizer only, not model weights) ──

@pytest.fixture(scope="session")
def real_tokenizer(cfg):
    """Load actual LLaDA tokenizer with config-specific special tokens."""
    from transformers import AutoTokenizer
    from src.model import added_tokens

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
    )

    # Base special tokens (always)
    special_tokens = (
        added_tokens.BOOL + added_tokens.FLOAT + added_tokens.DESCRIPTION
        + added_tokens.MOL_2D + added_tokens.MOL_3D
        + added_tokens.MOL_EMBEDDING + added_tokens.NUMBER
        + added_tokens.INSTRUCTION + added_tokens.REACTION_DIRECTION
        + added_tokens.IUPAC + added_tokens.MOLFORMULA
    )

    # Mol representation tag (config-driven, one of)
    mol_token_type = cfg.tokenizer.mol_token_type
    if mol_token_type == "selfies":
        special_tokens += added_tokens.SELFIES
    elif mol_token_type == "smiles":
        special_tokens += added_tokens.SMILES

    tokenizer.add_tokens(special_tokens)
    return tokenizer


# ── Toy dataset fixtures ──

@pytest.fixture(scope="session")
def toy_train_dataset(cfg):
    """Load train dataset from config-specified path."""
    from src.data.dataset import MoleculeDataset
    return MoleculeDataset(os.path.join(cfg.data.root, cfg.data.splits.train))


@pytest.fixture(scope="session")
def toy_train_samples(toy_train_dataset):
    """Load first 10 samples from train dataset."""
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
