# Graph augmentation statistical verification (Phase 2.5)

Source: `dataset/Processed/raw_v1_10x_rephrase/Train` (first 1,000 samples)

Adapter: src/dataset_generation/graph_aug.py (max_retry=8)

## Pass criteria (from plan)

- atom_change_ratio median ∈ [0.20, 0.45]
- success_ratio ≥ 90%
- Tanimoto median ∈ [0.4, 0.85]
- degenerate (atom < 4) sample count = 0

## Results

- total samples drawn: 1,000
- skipped (no input molecule, e.g. text2mol tasks): 191
- processable: 809
- **success on processable**: 5.81% (47 / 809)
- success overall: 4.70% (47 / 1,000)

**failure detail (within processable)**:
  - new_smiles_parse_fail: 683
  - below_min_atoms: 34
  - ccc_fallback: 23
  - extract_exception:DecoderError: 17
  - exotic_element: 5

**atom_change_ratio distribution**:
  atom_change_ratio
    n              : 47
    mean           : 0.4007
    median         : 0.3846
    std            : 0.1754
    p05 / p95      : 0.1033 / 0.6812
    min / max      : 0.0000 / 0.7778

**Tanimoto (MACCS) distribution** (chosen vs rejected):
  tanimoto
    n              : 47
    mean           : 0.3050
    median         : 0.2692
    std            : 0.2066
    p05 / p95      : 0.0749 / 0.6330
    min / max      : 0.0588 / 0.9600

**rejected molecule atom count distribution**:
  degenerate (< 4 atoms): 0
  median atoms: 7
  min / max:    4 / 20

**per-task success ratio (top 15)**:
```
    7.0%      16 /    230  smol-retrosynthesis
    1.3%       3 /    226  smol-forward_synthesis
    6.1%       4 /     66  chebi-20-mol2text
    7.7%       5 /     65  smol-name_conversion-s2i
    4.8%       3 /     63  smol-molecule_captioning
   13.2%       5 /     38  qm9_homo_lumo_gap
   12.5%       4 /     32  qm9_homo
   22.6%       7 /     31  qm9_lumo
    0.0%       0 /     27  reagent_prediction
    0.0%       0 /     11  forward_reaction_prediction
    0.0%       0 /      9  smol-property_prediction-hiv
    0.0%       0 /      6  smol-property_prediction-sider
    0.0%       0 /      3  smol-property_prediction-lipo
    0.0%       0 /      2  smol-property_prediction-bbbp
```

## Verdict

- atom_change_ratio median (0.385) in [0.20, 0.55]: PASS
- success_ratio on processable (5.81%) ≥ 50%: FAIL
- Tanimoto median (0.269) in [0.30, 0.70]: FAIL
- degenerate atom count == 0: PASS (0 found)

**OVERALL**: FAIL — tune adapter and re-run