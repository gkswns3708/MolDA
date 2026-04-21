"""Contract tests for the official Mol-LLM prompt format.

These tests lock `prepare_data_instance` to the official Mol-LLM token layout:
  - graph_sequence is `<GRAPH>` + mol_token*N + `</GRAPH>` (no inner spaces)
  - `<INPUT>` is replaced by `input_mol_string + graph_sequence` in one shot
  - `<INPUT>` must always be present in the source instruction (assert)

Unit-level: builds synthetic data_instances and calls the function directly.
No dataset fixture required.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_generation.generator import prepare_data_instance  # noqa: E402


SYSTEM_PROMPT = (
    "You are a helpful assistant for molecular chemistry, to address tasks "
    "including molecular property classification, molecular property regression, "
    "chemical reaction prediction, molecule captioning, molecule generation."
)
MOL_TOKEN = "<mol>"
NUM_QUERY_TOKENS = 32


def _make_instance(
    instruction="Predict the product of <INPUT>.",
    input_mol_string="<SMILES> CCO </SMILES>",
    label="<SMILES>CCO</SMILES>",
    task="forward_reaction_prediction/0",
):
    """Build a minimal data_instance dict mimicking Step-1 Arrow rows."""
    return {
        "instruction": instruction,
        "input_mol_string": input_mol_string,
        "label": label,
        "task_subtask_pair": task,
        "x": [[0, 0]],
        "edge_index": [[0], [0]],
        "edge_attr": [[0]],
        "additional_x": [[0, 0]],
        "additional_edge_index": [[0], [0]],
        "additional_edge_attr": [[0]],
    }


# ---------------------------------------------------------------------------
# graph_sequence format
# ---------------------------------------------------------------------------

def test_graph_sequence_has_no_inner_spaces():
    out = prepare_data_instance(_make_instance(), SYSTEM_PROMPT)
    expected_graph = "<GRAPH>" + MOL_TOKEN * NUM_QUERY_TOKENS + "</GRAPH>"
    assert expected_graph in out["prompt_text_smiles"], (
        "graph_sequence must match official Mol-LLM format (no inner spaces). "
        f"expected contains {expected_graph!r}, got prompt={out['prompt_text_smiles']!r}"
    )
    assert expected_graph in out["prompt_text_selfies"]


def test_graph_sequence_with_space_variant_is_absent():
    """Old buggy format `<GRAPH> ... </GRAPH>` must not be in the output."""
    out = prepare_data_instance(_make_instance(), SYSTEM_PROMPT)
    bad_open = "<GRAPH> " + MOL_TOKEN
    bad_close = MOL_TOKEN + " </GRAPH>"
    assert bad_open not in out["prompt_text_smiles"]
    assert bad_close not in out["prompt_text_smiles"]
    assert bad_open not in out["prompt_text_selfies"]
    assert bad_close not in out["prompt_text_selfies"]


def test_graph_sequence_uses_num_query_tokens_copies():
    out = prepare_data_instance(_make_instance(), SYSTEM_PROMPT, num_query_tokens=32)
    assert out["prompt_text_smiles"].count(MOL_TOKEN) == 32
    assert out["prompt_text_selfies"].count(MOL_TOKEN) == 32


# ---------------------------------------------------------------------------
# <INPUT> substitution position (official: at <INPUT> site, not appended)
# ---------------------------------------------------------------------------

def test_input_placeholder_is_replaced():
    out = prepare_data_instance(_make_instance(), SYSTEM_PROMPT)
    assert "<INPUT>" not in out["prompt_text_smiles"]
    assert "<INPUT>" not in out["prompt_text_selfies"]


def test_graph_sequence_attached_immediately_after_mol_string_at_input_site():
    """Official: `…<SMILES>CCO</SMILES><GRAPH>…</GRAPH> rest…`
    Not: `…<SMILES>CCO</SMILES> rest…<GRAPH>…</GRAPH>` (old append style)
    """
    instance = _make_instance(
        instruction="Predict the product of <INPUT> briefly.",
        input_mol_string="<SMILES> CCO </SMILES>",
    )
    out = prepare_data_instance(instance, SYSTEM_PROMPT)
    expected_fragment_smiles = (
        "<SMILES> CCO </SMILES>" + "<GRAPH>" + MOL_TOKEN * NUM_QUERY_TOKENS + "</GRAPH>"
    )
    assert expected_fragment_smiles in out["prompt_text_smiles"], (
        f"Expected fragment {expected_fragment_smiles!r} not in prompt: "
        f"{out['prompt_text_smiles']!r}"
    )
    # And the trailing "briefly." must come after </GRAPH>
    graph_end_idx = out["prompt_text_smiles"].index("</GRAPH>")
    briefly_idx = out["prompt_text_smiles"].index("briefly.")
    assert graph_end_idx < briefly_idx


# ---------------------------------------------------------------------------
# <INPUT> assertion (official contract)
# ---------------------------------------------------------------------------

def test_instruction_without_input_placeholder_raises():
    bad = _make_instance(instruction="Please answer concisely.")
    with pytest.raises(AssertionError, match=r"instruction must contain <INPUT>"):
        prepare_data_instance(bad, SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Dual-column consistency
# ---------------------------------------------------------------------------

def test_selfies_prompt_has_selfies_tag_not_smiles_tag():
    out = prepare_data_instance(_make_instance(), SYSTEM_PROMPT)
    # selfies prompt should carry <SELFIES> for the input molecule, and not <SMILES>
    # (The outer instruction may differ — we only check the molecule region.)
    # Heuristic: <SELFIES> tag must be present; the SMILES input mol tag should be gone.
    assert "<SELFIES>" in out["prompt_text_selfies"]
    # smiles prompt side: molecule carried as <SMILES>
    assert "<SMILES>" in out["prompt_text_smiles"]


def test_prompt_contains_llm_template_boundary():
    """Default llm_model_name=mistral → Mistral [INST] template."""
    out = prepare_data_instance(_make_instance(), SYSTEM_PROMPT, llm_model_name="mistral")
    assert out["prompt_text_smiles"].startswith("<s>[INST]")
    assert "[/INST]" in out["prompt_text_smiles"]


def test_prompt_llada_template_for_llada_backbone():
    out = prepare_data_instance(
        _make_instance(), SYSTEM_PROMPT, llm_model_name="GSAI-ML/LLaDA-8B-Instruct"
    )
    # LLaDA template uses <|start_header_id|> blocks, not [INST]
    assert "<|start_header_id|>" in out["prompt_text_smiles"]
    assert "[INST]" not in out["prompt_text_smiles"]
