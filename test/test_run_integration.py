"""Integration tests for the dataset generation pipeline.

Exercises the full pipeline end-to-end on a 2-task subset so we can verify:
  - step1/, step2/ directory layout
  - .step2_done marker behaviour (Step 2 skip on re-run)
  - --dedup off skips Step 2 and produces a distinct tag directory
  - Processed/{Train,Val,Test} final output contains the expected dual columns

Uses a synthetic config file pointing to bace only (fastest local task)
so network/HF downloads are minimal. Marked @pytest.mark.slow for opt-in.
"""

import os
import shutil
import sys
import tempfile

import datasets as hf_datasets
import pytest
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_generation.run import (  # noqa: E402
    STEP2_MARKER_FILENAME,
    _run_single_config_with_cfg,
    step1_dir,
    step2_dir,
    step2_marker,
    processed_dir,
    resolve_data_tag,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_pipeline_env(tmp_path):
    """Isolated raw/processed roots + a minimal single-task config.

    Uses bace because it reads from local BioT5 CSV fixtures in dataset/Raw/raw/
    and doesn't require network.
    """
    raw_root = tmp_path / "Raw"
    proc_root = tmp_path / "Processed"
    raw_root.mkdir()
    proc_root.mkdir()

    # bace requires BioT5 CSVs at {raw_data_root}/raw/BioT5_bace_{train,valid,test}.csv.
    # Symlink the existing project copies to keep this test lightweight.
    src_raw_dir = os.path.join(PROJECT_ROOT, "dataset", "Raw", "raw")
    if not os.path.isdir(src_raw_dir):
        pytest.skip("BACE CSV fixtures not found at dataset/Raw/raw/")
    (raw_root / "raw").symlink_to(src_raw_dir, target_is_directory=True)

    cfg = {
        "raw_data_root": str(raw_root),
        "processed_data_root": str(proc_root),
        "data_tag": "itest",
        "llm_model": "GSAI-ML/LLaDA-8B-Instruct",
        "target_benchmarks": ["bace"],
    }
    return {"cfg": cfg, "raw_root": str(raw_root), "proc_root": str(proc_root)}


# ---------------------------------------------------------------------------
# dedup on — step1/ + step2/ + marker + Processed/
# ---------------------------------------------------------------------------

def test_dedup_on_creates_step1_step2_marker_and_processed(tmp_pipeline_env):
    env = tmp_pipeline_env
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="on")

    tag = resolve_data_tag("itest", toy_n=5, dedup="on")
    s1 = step1_dir(env["raw_root"], tag)
    s2 = step2_dir(env["raw_root"], tag)
    marker = step2_marker(env["raw_root"], tag)
    p = processed_dir(env["proc_root"], tag)

    assert os.path.isdir(s1), f"step1/ missing: {s1}"
    assert os.path.isdir(s2), f"step2/ missing: {s2}"
    assert os.path.isfile(marker), f"step2 marker missing: {marker}"
    for split in ["Train", "Val", "Test"]:
        assert os.path.isdir(os.path.join(p, split)), f"Processed/{split}/ missing"


def test_dedup_on_processed_has_dual_columns(tmp_pipeline_env):
    env = tmp_pipeline_env
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="on")

    tag = resolve_data_tag("itest", toy_n=5, dedup="on")
    train_path = os.path.join(processed_dir(env["proc_root"], tag), "Train")
    ds = hf_datasets.Dataset.load_from_disk(train_path)
    assert len(ds) > 0
    for col in [
        "prompt_text_smiles",
        "prompt_text_selfies",
        "target_text_smiles",
        "target_text_selfies",
        "input_mol_string_smiles",
        "input_mol_string_selfies",
    ]:
        assert col in ds.column_names, f"missing dual column: {col}"


def test_dedup_on_processed_prompts_have_official_graph_format(tmp_pipeline_env):
    env = tmp_pipeline_env
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="on")

    tag = resolve_data_tag("itest", toy_n=5, dedup="on")
    train_path = os.path.join(processed_dir(env["proc_root"], tag), "Train")
    ds = hf_datasets.Dataset.load_from_disk(train_path)

    # 공식 포맷: 공백 없는 <GRAPH>...<mol>*32...</GRAPH>
    expected_graph = "<GRAPH>" + "<mol>" * 32 + "</GRAPH>"
    assert expected_graph in ds[0]["prompt_text_smiles"]
    # 옛 버그 포맷은 없어야 함
    assert "<GRAPH> <mol>" not in ds[0]["prompt_text_smiles"]


# ---------------------------------------------------------------------------
# dedup off — no step2/, no marker, distinct tag
# ---------------------------------------------------------------------------

def test_dedup_off_skips_step2_and_uses_nodedup_tag(tmp_pipeline_env):
    env = tmp_pipeline_env
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="off")

    tag = resolve_data_tag("itest", toy_n=5, dedup="off")
    assert tag.endswith("_nodedup"), f"unexpected tag: {tag}"

    s1 = step1_dir(env["raw_root"], tag)
    s2 = step2_dir(env["raw_root"], tag)
    marker = step2_marker(env["raw_root"], tag)
    p = processed_dir(env["proc_root"], tag)

    assert os.path.isdir(s1), f"step1/ missing under nodedup tag: {s1}"
    assert not os.path.isdir(s2), f"step2/ should NOT exist under dedup=off: {s2}"
    assert not os.path.isfile(marker), f"marker should NOT exist under dedup=off: {marker}"
    for split in ["Train", "Val", "Test"]:
        assert os.path.isdir(os.path.join(p, split))


# ---------------------------------------------------------------------------
# Re-run idempotency — .step2_done causes Step 2 to skip
# ---------------------------------------------------------------------------

def test_rerun_with_existing_marker_skips_step2(tmp_pipeline_env, capsys):
    env = tmp_pipeline_env
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="on")

    # Second run — should log the skip message
    capsys.readouterr()  # clear buffer
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="on")
    out = capsys.readouterr().out
    assert "Step 2] Skipped — marker exists" in out


# ---------------------------------------------------------------------------
# dedup on vs off produce isolated trees (no cross-pollination)
# ---------------------------------------------------------------------------

def test_dedup_on_and_off_coexist_in_separate_trees(tmp_pipeline_env):
    env = tmp_pipeline_env
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="on")
    _run_single_config_with_cfg(env["cfg"], "itest", num_workers=1, toy_n=5, dedup="off")

    on_tag = resolve_data_tag("itest", toy_n=5, dedup="on")
    off_tag = resolve_data_tag("itest", toy_n=5, dedup="off")
    assert on_tag != off_tag

    on_p = processed_dir(env["proc_root"], on_tag)
    off_p = processed_dir(env["proc_root"], off_tag)
    assert os.path.isdir(os.path.join(on_p, "Train"))
    assert os.path.isdir(os.path.join(off_p, "Train"))
