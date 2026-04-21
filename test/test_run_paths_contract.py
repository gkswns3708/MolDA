"""Contract tests for run.py path helpers.

Verifies deterministic directory layout:
  Raw/{data_tag}/step1/
  Raw/{data_tag}/step2/
  Raw/{data_tag}/.step2_done
  Processed/{data_tag}/{Train,Val,Test}/

And the dedup on/off → data_tag suffix convention.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_generation.run import (  # noqa: E402
    STEP2_MARKER_FILENAME,
    processed_dir,
    resolve_data_tag,
    step1_dir,
    step2_dir,
    step2_marker,
    task_arrow_name,
)


# ---------------------------------------------------------------------------
# resolve_data_tag: toy & nodedup suffixing
# ---------------------------------------------------------------------------

class TestResolveDataTag:
    def test_base_tag_unchanged_when_no_flags(self):
        assert resolve_data_tag("raw_v1", toy_n=None, dedup="on") == "raw_v1"

    def test_toy_suffix_applied(self):
        assert resolve_data_tag("raw_v1", toy_n=100, dedup="on") == "raw_v1_toy100"

    def test_nodedup_suffix_applied(self):
        assert resolve_data_tag("raw_v1", toy_n=None, dedup="off") == "raw_v1_nodedup"

    def test_toy_and_nodedup_compose(self):
        # toy → nodedup 순서로 붙어야 함 (일관성)
        assert resolve_data_tag("raw_v1", toy_n=100, dedup="off") == "raw_v1_toy100_nodedup"

    def test_zero_toy_n_treated_as_disabled(self):
        # toy_n=0은 falsy → toy suffix 미적용
        assert resolve_data_tag("raw_v1", toy_n=0, dedup="on") == "raw_v1"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

class TestDirHelpers:
    def test_step1_dir_path(self):
        assert step1_dir("/data/Raw", "raw_v1") == "/data/Raw/raw_v1/step1"

    def test_step2_dir_path(self):
        assert step2_dir("/data/Raw", "raw_v1") == "/data/Raw/raw_v1/step2"

    def test_step2_marker_is_sibling_of_step_dirs(self):
        marker = step2_marker("/data/Raw", "raw_v1")
        assert marker == f"/data/Raw/raw_v1/{STEP2_MARKER_FILENAME}"
        # marker는 step1/step2와 같은 부모에 있어야 함 (한 tag 단위로 재실행 skip 결정)
        assert os.path.dirname(marker) == os.path.dirname(step1_dir("/data/Raw", "raw_v1"))
        assert os.path.dirname(marker) == os.path.dirname(step2_dir("/data/Raw", "raw_v1"))

    def test_processed_dir_path(self):
        assert processed_dir("/data/Processed", "raw_v1") == "/data/Processed/raw_v1"


# ---------------------------------------------------------------------------
# Arrow directory naming
# ---------------------------------------------------------------------------

class TestArrowNaming:
    def test_basic_arrow_name(self):
        assert task_arrow_name("forward_reaction_prediction", 0, "train") == \
            "forward_reaction_prediction_subtask-0_train"

    def test_arrow_name_with_string_subtask(self):
        assert task_arrow_name("toxcast", "NR-AR", "test") == "toxcast_subtask-NR-AR_test"

    def test_all_three_splits_produce_distinct_names(self):
        names = {
            task_arrow_name("bace", 0, split) for split in ["train", "val", "test"]
        }
        assert len(names) == 3


# ---------------------------------------------------------------------------
# STEP2_MARKER_FILENAME is hidden (starts with dot)
# ---------------------------------------------------------------------------

def test_step2_marker_filename_is_dotfile():
    assert STEP2_MARKER_FILENAME.startswith(".")
    assert STEP2_MARKER_FILENAME == ".step2_done"


# ---------------------------------------------------------------------------
# Download configs declare processed_data_root alongside raw_data_root
# ---------------------------------------------------------------------------

class TestDownloadConfigContracts:
    @pytest.mark.parametrize("config_name", ["smiles", "selfies", "both"])
    def test_config_has_processed_data_root(self, config_name):
        import yaml
        config_path = os.path.join(
            PROJECT_ROOT, "src", "configs", "download", f"{config_name}.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "raw_data_root" in cfg
        assert "processed_data_root" in cfg, (
            f"{config_name}.yaml must declare processed_data_root for Step 3 output location"
        )
        assert cfg["processed_data_root"] != cfg["raw_data_root"], (
            "Raw and Processed roots must be distinct"
        )
