"""Benchmark task name constants for dataset generation and evaluation."""

CLASSIFICATION_BENCHMARKS = [
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
    "tox21",
    "toxcast",
]
REGRESSION_BENCHMARKS = [
    "smol-property_prediction-esol",
    "smol-property_prediction-lipo",
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap",
    "qm9_dipole_moment",
    "qm9_isotropic_polarizability",
    "qm9_electronic_spatial_extent",
    "qm9_zero_point_vibrational_energy",
    "qm9_heat_capacity_298K",
    "qm9_internal_energy_298K",
    "qm9_enthalpy_298K",
    "qm9_free_energy_298K",
    "alchemy_homo",
    "alchemy_lumo",
    "alchemy_homo_lumo_gap",
    "aqsol-logS",
    "pcqm_homo_lumo_gap",
]
REACTION_BENCHMARKS = [
    "forward_reaction_prediction",
    "smol-forward_synthesis",
    "retrosynthesis",
    "smol-retrosynthesis",
    "reagent_prediction",
    "presto-forward_reaction_prediction",
    "presto-retrosynthesis",
    "presto-reagent_prediction",
    "orderly-forward_reaction_prediction",
    "orderly-retrosynthesis",
    "orderly-reagent_prediction",
]
TEXT2MOL_BENCHMARKS = [
    "chebi-20-text2mol",
    "smol-molecule_generation",
]
MOL2TEXT_BENCHMARKS = [
    "chebi-20-mol2text",
    "smol-molecule_captioning",
]
NAME_CONVERSION_BENCHMARKS = [
    "smol-name_conversion-i2s",
    "smol-name_conversion-i2f",
    "smol-name_conversion-s2f",
    "smol-name_conversion-s2i",
]

ALL_BENCHMARKS = (
    CLASSIFICATION_BENCHMARKS
    + REGRESSION_BENCHMARKS
    + REACTION_BENCHMARKS
    + TEXT2MOL_BENCHMARKS
    + MOL2TEXT_BENCHMARKS
    + NAME_CONVERSION_BENCHMARKS
)
