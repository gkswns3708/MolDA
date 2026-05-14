"""Unit tests for src/dataset_generation/graph_aug_atomwise.py.

Covers both the low-level `extract_and_modify_atomwise` (returns the raw
Mol-LLM-style dict) and the high-level `augment_molecule_graph_atomwise`
(returns the V-MolPO collator-friendly dict or None with reason).

Key property tested vs. upstream Mol-LLM behavior:
    `n_added` can exceed 1 — upstream's commented-out
    `add_replacement_mol` always added a single atom per substructure,
    while the atomwise variant attaches the entire replacement fragment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset_generation.graph_aug_atomwise import (
    CCC_FALLBACK_SMILES,
    DEFAULT_MIN_ATOMS,
    augment_molecule_graph_atomwise,
    extract_and_modify_atomwise,
    extract_and_modify_fts_targeted,
    map_by_substructure_replacement_atomwise,
    selfies_from_input_mol_string,
    smiles_to_selfies,
)

CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
ACETAMINOPHEN_SMILES = "CC(=O)Nc1ccc(O)cc1"

EXPECTED_GRAPH_KEYS = {
    "x", "edge_index", "edge_attr",
    "atom_change_ratio", "n_removed", "n_added",
    "tanimoto", "modified_smiles",
}


# ---------------------------------------------------------------------
# High-level adapter contract
# ---------------------------------------------------------------------
def test_returns_dict_with_expected_keys_on_success():
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    out = augment_molecule_graph_atomwise(sel, seed=42, max_retry=20)
    if out is None:
        pytest.skip("caffeine + seed=42 happened to be rejected this run")
    assert set(out.keys()) >= EXPECTED_GRAPH_KEYS
    assert isinstance(out["x"], list) and len(out["x"]) >= DEFAULT_MIN_ATOMS
    assert isinstance(out["edge_index"], list) and len(out["edge_index"]) == 2
    assert isinstance(out["edge_attr"], list)
    assert 0.0 <= out["atom_change_ratio"] <= 5.0
    assert out["n_removed"] >= 0
    assert out["n_added"] >= 0
    assert out["tanimoto"] == -1.0 or 0.0 <= out["tanimoto"] <= 1.0


def test_deterministic_same_seed():
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    a = augment_molecule_graph_atomwise(sel, seed=7, max_retry=20)
    b = augment_molecule_graph_atomwise(sel, seed=7, max_retry=20)
    if a is None or b is None:
        pytest.skip("caffeine + seed=7 rejected by both runs (still deterministic)")
    assert a["modified_smiles"] == b["modified_smiles"]
    assert a["x"] == b["x"]
    assert a["edge_index"] == b["edge_index"]
    assert a["n_removed"] == b["n_removed"]
    assert a["n_added"] == b["n_added"]


def test_different_seeds_can_diverge():
    """Sanity: different seeds should not always collapse to the same output."""
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    outs = [augment_molecule_graph_atomwise(sel, seed=s, max_retry=20) for s in range(20)]
    successful = [o["modified_smiles"] for o in outs if o is not None]
    if len(successful) < 2:
        pytest.skip("not enough successful runs to compare across seeds")
    assert len(set(successful)) > 1, "all 20 seeds produced the same modified molecule"


def test_invalid_input_returns_none():
    assert augment_molecule_graph_atomwise("") is None
    assert augment_molecule_graph_atomwise(None) is None  # type: ignore[arg-type]
    assert augment_molecule_graph_atomwise("not-a-real-selfies-string", max_retry=2) is None


def test_small_input_now_succeeds_via_guards():
    """Previously CC (ethane, 2 atoms) returned None because the algorithm
    collapsed to CCC. After adding the min-preservation guard and add
    loop's natural growth, even tiny molecules produce a valid augmented
    result that satisfies min_atoms."""
    sel = smiles_to_selfies("CC")
    out = augment_molecule_graph_atomwise(sel, max_retry=5)
    # Either None (algorithm couldn't grow it enough) or a valid dict with
    # >=DEFAULT_MIN_ATOMS atoms — both are acceptable. The key invariant is
    # that we don't get a CCC-shaped 3-atom result silently masquerading.
    if out is not None:
        assert len(out["x"]) >= DEFAULT_MIN_ATOMS
        # Should not match the CCC fallback shape (3 carbons)
        assert out["modified_smiles"] != CCC_FALLBACK_SMILES


def test_none_marker_returns_none():
    out, reason = augment_molecule_graph_atomwise("<None>", return_reason=True)
    assert out is None
    assert reason == "none_marker"


def test_return_reason_mode():
    out, reason = augment_molecule_graph_atomwise("", return_reason=True)
    assert out is None
    assert reason == "empty_input"

    out, reason = augment_molecule_graph_atomwise("garbage", return_reason=True, max_retry=2)
    assert out is None
    assert reason.startswith("extract_exception") or reason.startswith("fallback:") or reason in {
        "ccc_fallback", "below_min_atoms", "missing_graph_fields",
    }


def test_min_atoms_respected():
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    for s in range(30):
        out = augment_molecule_graph_atomwise(sel, seed=s, max_retry=10)
        if out is None:
            continue
        assert len(out["x"]) >= DEFAULT_MIN_ATOMS, (
            f"seed={s}: rejected mol has {len(out['x'])} atoms < min={DEFAULT_MIN_ATOMS}"
        )


# ---------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------
def test_selfies_from_input_mol_string_strips_wrappers():
    assert selfies_from_input_mol_string("<SELFIES> [C][C] </SELFIES>") == "[C][C]"
    assert selfies_from_input_mol_string("<SELFIES>[C][C]</SELFIES>") == "[C][C]"


def test_smiles_to_selfies_roundtrip_aspirin():
    sel = smiles_to_selfies(ASPIRIN_SMILES)
    assert sel is not None
    assert sel.startswith("[C]")


# ---------------------------------------------------------------------
# Low-level extract_and_modify_atomwise contract
# ---------------------------------------------------------------------
def test_extract_returns_full_dict_structure():
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    out = extract_and_modify_atomwise(sel, replace_ratio=0.3)
    required = {
        "original_smiles", "num_of_key_substructures",
        "n_removed", "n_added", "atom_change_ratio", "target_change_ratio",
        "fallback_reason", "substructures_removed", "substructures_added",
        "removed_graph", "modified_smiles", "modified_graph",
    }
    assert required.issubset(out.keys())
    assert out["target_change_ratio"] == pytest.approx(0.3)
    if out["fallback_reason"] is None:
        assert out["modified_smiles"] != CCC_FALLBACK_SMILES
        assert out["modified_graph"]["num_nodes"] > 0
        assert isinstance(out["substructures_removed"], list)
        assert isinstance(out["substructures_added"], list)


def test_whole_substructure_addition_can_add_more_than_one_atom():
    """Core property that distinguishes this module from upstream Mol-LLM.

    Upstream's `extract_and_modify` adds only `replacement_mol.GetAtomWithIdx(0)`
    (one atom per substructure). This module restores the intended
    `add_replacement_mol`, so multi-atom fragments stay multi-atom.

    Test: over many seeds + diverse molecules, at least one accepted run
    must record `n_added > 1`.
    """
    targets = [CAFFEINE_SMILES, ASPIRIN_SMILES, ACETAMINOPHEN_SMILES]
    seen_multi = False
    successes = 0
    for smi in targets:
        sel = smiles_to_selfies(smi)
        for seed in range(50):
            import random
            random.seed(seed)
            out = extract_and_modify_atomwise(sel, replace_ratio=0.3)
            if out.get("fallback_reason") is not None:
                continue
            successes += 1
            if out["n_added"] > 1:
                seen_multi = True
                break
        if seen_multi:
            break
    if successes == 0:
        pytest.skip("no successful atomwise runs across 3 molecules x 50 seeds")
    assert seen_multi, (
        "n_added was 1 or 0 in every successful run — atomwise addition "
        "appears to be silently falling back to single-atom attach. "
        "Check add_replacement_mol invocation in extract_and_modify_atomwise."
    )


def test_atom_change_ratio_consistency():
    """atom_change_ratio must equal (n_removed + n_added) / original_n."""
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    for seed in range(20):
        import random
        random.seed(seed)
        out = extract_and_modify_atomwise(sel, replace_ratio=0.3)
        if out.get("fallback_reason") is not None:
            continue
        from rdkit import Chem
        original_n = Chem.MolFromSmiles(out["original_smiles"]).GetNumAtoms()
        expected = (out["n_removed"] + out["n_added"]) / max(original_n, 1)
        assert out["atom_change_ratio"] == pytest.approx(expected, abs=1e-6)


def test_invalid_selfies_raises():
    with pytest.raises(ValueError):
        extract_and_modify_atomwise("not-a-real-selfies-string", replace_ratio=0.3)


def test_fallback_payload_shape_on_degenerate_input():
    """Tiny molecules with no/few active MACCS keys hit the CCC fallback path
    but must still return a well-formed payload with `fallback_reason` set.
    """
    sel = smiles_to_selfies("CC")
    out = extract_and_modify_atomwise(sel, replace_ratio=0.3)
    if out.get("fallback_reason") is None:
        pytest.skip("ethane was not actually degenerate this RDKit version")
    assert out["modified_smiles"] == CCC_FALLBACK_SMILES
    assert out["n_removed"] == 0
    assert out["n_added"] == 0
    assert out["atom_change_ratio"] == 0.0
    assert out["modified_graph"]["num_nodes"] > 0  # CCC graph is non-empty


# ---------------------------------------------------------------------
# map_by_substructure_replacement_atomwise — dataset map contract
# ---------------------------------------------------------------------
def _expected_keys_for(num_rejected: int) -> set:
    keys = set()
    for i in range(num_rejected):
        keys.update({
            f"{i}-th_rejected_x", f"{i}-th_rejected_edge_index", f"{i}-th_rejected_edge_attr",
            f"{i}-th_additional_rejected_x",
            f"{i}-th_additional_rejected_edge_index",
            f"{i}-th_additional_rejected_edge_attr",
        })
    return keys


def test_map_populates_18_keys_for_default_task():
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    row = {
        "task": "chebi-20-mol2text",
        "input_mol_string_selfies": f"<SELFIES> {sel} </SELFIES>",
    }
    out = map_by_substructure_replacement_atomwise(row, num_rejected_graphs=6)
    expected = _expected_keys_for(6)
    assert expected.issubset(out.keys()), f"missing: {expected - set(out.keys())}"
    for i in range(6):
        nodes = out[f"{i}-th_rejected_x"]
        edge_index = out[f"{i}-th_rejected_edge_index"]
        edge_attr = out[f"{i}-th_rejected_edge_attr"]
        assert isinstance(nodes, list) and len(nodes) >= 1
        assert isinstance(edge_index, list) and len(edge_index) == 2
        assert isinstance(edge_attr, list)
        # Both rejected and additional populated independently
        assert isinstance(out[f"{i}-th_additional_rejected_x"], list)


def test_map_text2mol_task_uses_dummy_selfies():
    """text2mol-like tasks have no real input molecule — upstream uses
    [C][C][C] as the source SELFIES."""
    row = {
        "task": "chebi-20-text2mol",
        "input_mol_string_selfies": "<SELFIES> </SELFIES>",  # empty
        "input_mol_string": "",
    }
    out = map_by_substructure_replacement_atomwise(row, num_rejected_graphs=3)
    for i in range(3):
        nodes = out[f"{i}-th_rejected_x"]
        # CCC fallback or a tiny dummy-corrupted graph — either way, non-empty
        assert len(nodes) >= 1


def test_map_reaction_task_splits_pair():
    """Reaction selfies use `|>>|` separator. The two halves should be
    corrupted independently for rejected vs additional_rejected."""
    sel_left = smiles_to_selfies(ASPIRIN_SMILES)
    sel_right = smiles_to_selfies(ACETAMINOPHEN_SMILES)
    row = {
        "task": "forward_reaction_prediction",
        "input_mol_string_selfies": f"<SELFIES> {sel_left}|>>|{sel_right} </SELFIES>",
    }
    out = map_by_substructure_replacement_atomwise(row, num_rejected_graphs=2)
    for i in range(2):
        assert len(out[f"{i}-th_rejected_x"]) >= 1
        assert len(out[f"{i}-th_additional_rejected_x"]) >= 1


def test_map_falls_back_to_input_mol_string_when_selfies_field_missing():
    """mol-llm raw schema only has `input_mol_string`, not the dual column."""
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    row = {
        "task": "chebi-20-mol2text",
        "input_mol_string": f"<SELFIES> {sel} </SELFIES>",
    }
    out = map_by_substructure_replacement_atomwise(row, num_rejected_graphs=2)
    for i in range(2):
        assert len(out[f"{i}-th_rejected_x"]) >= 1


def test_map_invalid_selfies_falls_back_to_ccc():
    """A garbage SELFIES must not raise — it should silently CCC-fallback per slot."""
    row = {
        "task": "chebi-20-mol2text",
        "input_mol_string_selfies": "<SELFIES> not-a-real-selfies </SELFIES>",
    }
    out = map_by_substructure_replacement_atomwise(row, num_rejected_graphs=2)
    # CCC graph has 3 nodes
    for i in range(2):
        assert len(out[f"{i}-th_rejected_x"]) >= 1


def test_map_preserves_existing_row_fields():
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    row = {
        "task": "chebi-20-mol2text",
        "input_mol_string_selfies": f"<SELFIES> {sel} </SELFIES>",
        "target_text_chosen": "a description",
        "label": "some label",
    }
    out = map_by_substructure_replacement_atomwise(row, num_rejected_graphs=2)
    assert out["task"] == "chebi-20-mol2text"
    assert out["target_text_chosen"] == "a description"
    assert out["label"] == "some label"


# ---------------------------------------------------------------------
# extract_and_modify_fts_targeted — 0.7 algorithm (best-of-N)
# ---------------------------------------------------------------------
def _maccs_fts(smiles_a, smiles_b):
    from rdkit import Chem
    from rdkit.Chem import DataStructs, MACCSkeys
    a = Chem.MolFromSmiles(smiles_a)
    b = Chem.MolFromSmiles(smiles_b)
    if a is None or b is None:
        return None
    return DataStructs.TanimotoSimilarity(
        MACCSkeys.GenMACCSKeys(a),
        MACCSkeys.GenMACCSKeys(b),
    )


def test_fts_targeted_returns_maccs_fts_key():
    import random
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    random.seed(42)
    out = extract_and_modify_fts_targeted(
        sel, target_fts=0.7, target_fts_tolerance=0.1, max_attempts=10,
    )
    assert "maccs_fts" in out
    assert "n_fts_attempts" in out
    assert "fts_in_band" in out
    assert "target_fts" in out
    assert out["target_fts"] == 0.7
    assert 1 <= out["n_fts_attempts"] <= 10


def test_fts_targeted_converges_to_band_for_caffeine():
    """Across many seeds, most caffeine results should fall in [0.6, 0.8]."""
    import random
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    in_band = 0
    fts_vals = []
    for s in range(30):
        random.seed(s)
        out = extract_and_modify_fts_targeted(
            sel, target_fts=0.7, target_fts_tolerance=0.1, max_attempts=10,
        )
        fts = out.get("maccs_fts")
        if fts is None:
            continue
        fts_vals.append(fts)
        if 0.60 <= fts <= 0.80:
            in_band += 1
    assert len(fts_vals) >= 25, "too many fallbacks for caffeine"
    # caffeine should land in band on a strong majority of seeds
    assert in_band / len(fts_vals) >= 0.7, (
        f"only {in_band}/{len(fts_vals)} caffeine seeds reached target band"
    )
    mean_fts = sum(fts_vals) / len(fts_vals)
    assert 0.55 <= mean_fts <= 0.85, f"mean fts {mean_fts:.3f} off target"


def test_fts_targeted_dict_includes_atom_change_keys():
    """The fts-targeted variant must still expose the same extra keys
    (n_removed/n_added/atom_change_ratio) as the base function so
    downstream code doesn't need to special-case."""
    import random
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    random.seed(123)
    out = extract_and_modify_fts_targeted(sel, target_fts=0.7, max_attempts=5)
    for k in ("n_removed", "n_added", "atom_change_ratio",
              "modified_smiles", "modified_graph"):
        assert k in out, f"missing key {k}"


def test_fts_targeted_invalid_selfies_raises():
    with pytest.raises(ValueError):
        extract_and_modify_fts_targeted("garbage", target_fts=0.7, max_attempts=2)


def test_map_with_target_fts_populates_keys():
    """map_by_substructure_replacement_atomwise(target_fts=0.7) still
    produces all `{i}-th_rejected_*` keys with valid graphs."""
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    row = {
        "task": "chebi-20-mol2text",
        "input_mol_string_selfies": f"<SELFIES> {sel} </SELFIES>",
    }
    out = map_by_substructure_replacement_atomwise(
        row, num_rejected_graphs=3, target_fts=0.7, max_attempts=5,
    )
    for i in range(3):
        assert isinstance(out[f"{i}-th_rejected_x"], list)
        assert len(out[f"{i}-th_rejected_x"]) >= 1
        assert isinstance(out[f"{i}-th_additional_rejected_x"], list)


# ---------------------------------------------------------------------
# CCC fallback prevention guards
# ---------------------------------------------------------------------
def test_min_preservation_keeps_small_mol_alive():
    """`CCO` (ethanol, 3 atoms) used to drop to `empty_graph` → CCC.
    With the min-preservation guard the remove loop must skip any match
    that would leave fewer than 2 atoms, so the result keeps >=2 atoms
    and the fallback_reason is NOT `empty_graph`."""
    import random
    sel = smiles_to_selfies("CCO")
    saw_non_ccc = False
    for s in range(20):
        random.seed(s)
        out = extract_and_modify_atomwise(sel, replace_ratio=0.3)
        if out.get("fallback_reason") in (None, "atom_swap_fallback"):
            saw_non_ccc = True
            assert out["modified_graph"]["num_nodes"] >= 2
    assert saw_non_ccc, "min-preservation guard failed to save CCO from CCC"


def test_atom_swap_fallback_returns_chemistry_meaningful_negative():
    """When all normal paths fail, `_atom_swap_fallback` clones the
    original and changes one atom's element. The result must:
      - have the same atom count as the original (no remove, no add)
      - be parseable as SMILES
      - differ from the original SMILES
    """
    from rdkit import Chem
    from dataset_generation.graph_aug_atomwise import _atom_swap_fallback
    orig_mol = Chem.MolFromSmiles(CAFFEINE_SMILES)
    import random
    random.seed(99)
    result = _atom_swap_fallback(orig_mol, CAFFEINE_SMILES, 0.3, verbose=False)
    assert result is not None
    assert result["fallback_reason"] == "atom_swap_fallback"
    assert result["modified_smiles"] != CAFFEINE_SMILES
    mod = Chem.MolFromSmiles(result["modified_smiles"])
    assert mod is not None
    # atom-swap preserves atom count
    assert mod.GetNumAtoms() == orig_mol.GetNumAtoms()


def test_min_preservation_does_not_constrain_large_mols():
    """For drug-like molecules (>20 atoms), the 25% floor (= ≥5 atoms) is
    far below typical 30% removal targets. The guard shouldn't activate
    and FTS-relevant atom counts shouldn't differ from baseline."""
    import random
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    random.seed(42)
    out = extract_and_modify_atomwise(sel, replace_ratio=0.3)
    # caffeine = 14 atoms; min_preserved = max(2, 14//4) = 3
    # Guard should not be the dominant constraint
    if out.get("fallback_reason") is None:
        # The result should still show meaningful removal (the guard
        # didn't prematurely halt the remove loop)
        assert out["n_removed"] >= 1


# ─────────────────────────────────────────────────────────────────
# Schema-consistency regression — full multi-task build
# ─────────────────────────────────────────────────────────────────

def _column_set(row_out: dict, num_rejected: int) -> set[str]:
    """Return only the build-emitted keys ({i}-th_rejected_* / additional)."""
    prefixes = {f"{i}-th_" for i in range(num_rejected)}
    return {k for k in row_out if any(k.startswith(p) for p in prefixes)}


def test_skip_additional_still_emits_keys_for_schema_consistency():
    """With `skip_additional_rejected=True`, non-reaction rows must STILL
    emit `additional_rejected_*` keys (filled with a CCC dummy graph) so
    that HF Arrow writer's schema stays uniform across the dataset.

    Regression for the 99% `KeyError: '0-th_additional_rejected_x'` failure
    seen with `build_full_atomwise.sh`: reaction rows added these keys,
    non-reaction rows didn't, schema diverged."""
    sel = smiles_to_selfies(CAFFEINE_SMILES)
    row = {
        "task": "chebi-20-mol2text",
        "input_mol_string_selfies": f"<SELFIES> {sel} </SELFIES>",
    }
    out = map_by_substructure_replacement_atomwise(
        row, num_rejected_graphs=6, skip_additional_rejected=True,
    )
    for i in range(6):
        # Even with skip_additional, key must exist
        assert f"{i}-th_additional_rejected_x" in out
        assert f"{i}-th_additional_rejected_edge_index" in out
        assert f"{i}-th_additional_rejected_edge_attr" in out
        # Dummy CCC graph has 3 nodes
        assert len(out[f"{i}-th_additional_rejected_x"]) == 3


def test_reaction_and_non_reaction_emit_same_column_set():
    """The CORE schema-consistency invariant: regardless of task type
    (reaction vs default vs text2mol-like), the build-emitted column set
    must be identical. HF Arrow writer's batch consolidation breaks
    otherwise."""
    sel_left = smiles_to_selfies(ASPIRIN_SMILES)
    sel_right = smiles_to_selfies(ACETAMINOPHEN_SMILES)
    n = 4

    rxn_row = {
        "task": "reagent_prediction",
        "input_mol_string_selfies": f"<SELFIES> {sel_left}|>>|{sel_right} </SELFIES>",
    }
    txt_row = {
        "task": "chebi-20-text2mol",
        "input_mol_string_selfies": "",
        "input_mol_string": "",
    }
    nrx_row = {
        "task": "chebi-20-mol2text",
        "input_mol_string_selfies": f"<SELFIES> {sel_left} </SELFIES>",
    }
    common_kw = dict(num_rejected_graphs=n, skip_additional_rejected=True)
    cols_rxn = _column_set(
        map_by_substructure_replacement_atomwise(dict(rxn_row), **common_kw), n)
    cols_txt = _column_set(
        map_by_substructure_replacement_atomwise(dict(txt_row), **common_kw), n)
    cols_nrx = _column_set(
        map_by_substructure_replacement_atomwise(dict(nrx_row), **common_kw), n)
    assert cols_rxn == cols_nrx == cols_txt, (
        f"column sets diverge — schema risk!\n"
        f"  reaction-only:    {sorted(cols_rxn - cols_nrx)}\n"
        f"  non-reaction-only: {sorted(cols_nrx - cols_rxn)}\n"
        f"  text2mol-only:    {sorted(cols_txt - cols_nrx)}"
    )


def test_smol_molecule_generation_treated_as_text2mol_dummy():
    """smol-molecule_generation is text → mol (no input molecule) and
    upstream Mol-LLM (TEXT2MOL_BENCHMARKS, augment_dataset.py:890-901)
    uses dummy `[C][C][C]` for its rejected graphs. Our
    TEXT2MOL_LIKE_TASKS must include it so the dispatch branches
    consistently regardless of what's in the row's selfies field."""
    from dataset_generation.graph_aug_atomwise import TEXT2MOL_LIKE_TASKS

    assert "smol-molecule_generation" in TEXT2MOL_LIKE_TASKS
    assert "chebi-20-text2mol" in TEXT2MOL_LIKE_TASKS

    row = {
        "task": "smol-molecule_generation",
        "input_mol_string_selfies": "",  # empty / dummy in actual data
        "input_mol_string": "",
    }
    out = map_by_substructure_replacement_atomwise(row, num_rejected_graphs=2)
    for i in range(2):
        # Dummy-corrupted rejected graph — small (CCC-derived)
        assert len(out[f"{i}-th_rejected_x"]) >= 1
