"""Atomwise graph augmentation for V-MolPO rejected pairs.

Self-contained variant of Mol-LLM's MACCS-based substructure replacement
(upstream `extract_and_modify` in /opt/mol-llm_official/augment_dataset.py).

Only the addition step differs from upstream: upstream commented out
`add_replacement_mol(copy_mol, replacement_mol, attach_idx)` (L610) and
replaced it with a 1-atom dummy attach (L611-L614); this module restores
the intended whole-substructure attach plus a real snapshot/sanitize
guard. See docs/GRAPH_AUG_ATOMWISE.md for the full algorithm and
rationale.

Utilities (smartsPatts, atom_groups, make_specific_smarts,
add_replacement_mol, mol2graph, atom/bond feature helpers) are inlined
verbatim from the Mol-LLM upstream so this module has no dependency on
the deleted graph_aug_legacy.
"""
from __future__ import annotations

import random
import re
import sys

import numpy as np
import selfies as sf
from ogb.utils.features import allowable_features
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, MACCSkeys

# Suppress RDKit's verbose valence/kekulize warnings — atomwise augmentation
# intentionally tries many invalid valence combinations and relies on
# try/except. With num_proc=48 on 4M rows the stderr output reaches 5+ GB
# and can OOM-kill the parent process via stdout pipe buffer pressure.
RDLogger.DisableLog("rdApp.*")


# =====================================================================
# MACCS SMARTS dictionary — verbatim from Mol-LLM augment_dataset.py
# =====================================================================
smartsPatts = {
    2: ("[#104]", 0),
    3: ("[#32,#33,#34,#50,#51,#52,#82,#83,#84]", 0),
    4: ("[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]", 0),
    5: ("[Sc,Ti,Y,Zr,Hf]", 0),
    6: ("[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]", 0),
    7: ("[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]", 0),
    8: ("[!#6;!#1]1~*~*~*~1", 0),
    9: ("[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]", 0),
    10: ("[Be,Mg,Ca,Sr,Ba,Ra]", 0),
    11: ("*1~*~*~*~1", 0),
    12: ("[Cu,Zn,Ag,Cd,Au,Hg]", 0),
    13: ("[#8]~[#7](~[#6])~[#6]", 0),
    14: ("[#16]-[#16]", 0),
    15: ("[#8]~[#6](~[#8])~[#8]", 0),
    16: ("[!#6;!#1]1~*~*~1", 0),
    17: ("[#6]#[#6]", 0),
    18: ("[#5,#13,#31,#49,#81]", 0),
    19: ("*1~*~*~*~*~*~*~1", 0),
    20: ("[#14]", 0),
    21: ("[#6]=[#6](~[!#6;!#1])~[!#6;!#1]", 0),
    22: ("*1~*~*~1", 0),
    23: ("[#7]~[#6](~[#8])~[#8]", 0),
    24: ("[#7]-[#8]", 0),
    25: ("[#7]~[#6](~[#7])~[#7]", 0),
    26: ("[#6]=;@[#6](@*)@*", 0),
    27: ("[I]", 0),
    28: ("[!#6;!#1]~[CH2]~[!#6;!#1]", 0),
    29: ("[#15]", 0),
    30: ("[#6]~[!#6;!#1](~[#6])(~[#6])~*", 0),
    31: ("[!#6;!#1]~[F,Cl,Br,I]", 0),
    32: ("[#6]~[#16]~[#7]", 0),
    33: ("[#7]~[#16]", 0),
    34: ("[CH2]=*", 0),
    35: ("[Li,Na,K,Rb,Cs,Fr]", 0),
    36: ("[#16R]", 0),
    37: ("[#7]~[#6](~[#8])~[#7]", 0),
    38: ("[#7]~[#6](~[#6])~[#7]", 0),
    39: ("[#8]~[#16](~[#8])~[#8]", 0),
    40: ("[#16]-[#8]", 0),
    41: ("[#6]#[#7]", 0),
    43: ("[!#6;!#1;!H0]~*~[!#6;!#1;!H0]", 0),
    45: ("[#6]=[#6]~[#7]", 0),
    47: ("[#16]~*~[#7]", 0),
    48: ("[#8]~[!#6;!#1](~[#8])(~[#8])", 0),
    50: ("[#6]=[#6](~[#6])~[#6]", 0),
    51: ("[#6]~[#16]~[#8]", 0),
    52: ("[#7]~[#7]", 0),
    53: ("[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]", 0),
    54: ("[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]", 0),
    55: ("[#8]~[#16]~[#8]", 0),
    56: ("[#8]~[#7](~[#8])~[#6]", 0),
    57: ("[#8R]", 0),
    58: ("[!#6;!#1]~[#16]~[!#6;!#1]", 0),
    60: ("[#16]=[#8]", 0),
    61: ("*~[#16](~*)~*", 0),
    62: ("*@*!@*@*", 0),
    63: ("[#7]=[#8]", 0),
    64: ("*@*!@[#16]", 0),
    66: ("[#6]~[#6](~[#6])(~[#6])~*", 0),
    67: ("[!#6;!#1]~[#16]", 0),
    68: ("[!#6;!#1;!H0]~[!#6;!#1;!H0]", 0),
    69: ("[!#6;!#1]~[!#6;!#1;!H0]", 0),
    70: ("[!#6;!#1]~[#7]~[!#6;!#1]", 0),
    71: ("[#7]~[#8]", 0),
    72: ("[#8]~*~*~[#8]", 0),
    73: ("[#16]=*", 0),
    74: ("[CH3]~*~[CH3]", 0),
    75: ("*!@[#7]@*", 0),
    76: ("[#6]=[#6](~*)~*", 0),
    77: ("[#7]~*~[#7]", 0),
    78: ("[#6]=[#7]", 0),
    79: ("[#7]~*~*~[#7]", 0),
    80: ("[#7]~*~*~*~[#7]", 0),
    81: ("[#16]~*(~*)~*", 0),
    82: ("*~[CH2]~[!#6;!#1;!H0]", 0),
    83: ("[!#6;!#1]1~*~*~*~*~1", 0),
    84: ("[NH2]", 0),
    85: ("[#6]~[#7](~[#6])~[#6]", 0),
    86: ("[C;H2,H3][!#6;!#1][C;H2,H3]", 0),
    87: ("[F,Cl,Br,I]!@*@*", 0),
    88: ("[#16]", 0),
    89: ("[#8]~*~*~*~[#8]", 0),
    92: ("[#8]~[#6](~[#7])~[#6]", 0),
    93: ("[!#6;!#1]~[CH3]", 0),
    94: ("[!#6;!#1]~[#7]", 0),
    95: ("[#7]~*~*~[#8]", 0),
    96: ("*1~*~*~*~*~1", 0),
    97: ("[#7]~*~*~*~[#8]", 0),
    98: ("[!#6;!#1]1~*~*~*~*~*~1", 0),
    99: ("[#6]=[#6]", 0),
    100: ("*~[CH2]~[#7]", 0),
    102: ("[!#6;!#1]~[#8]", 0),
    104: ("[!#6;!#1;!H0]~*~[CH2]~*", 0),
    105: ("*@*(@*)@*", 0),
    106: ("[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]", 0),
    107: ("[F,Cl,Br,I]~*(~*)~*", 0),
    108: ("[CH3]~*~*~*~[CH2]~*", 0),
    109: ("*~[CH2]~[#8]", 0),
    110: ("[#7]~[#6]~[#8]", 0),
    111: ("[#7]~*~[CH2]~*", 0),
    112: ("*~*(~*)(~*)~*", 0),
    114: ("[CH3]~[CH2]~*", 0),
    115: ("[CH3]~*~[CH2]~*", 0),
    117: ("[#7]~*~[#8]", 0),
    119: ("[#7]=*", 0),
    122: ("*~[#7](~*)~*", 0),
    123: ("[#8]~[#6]~[#8]", 0),
    124: ("[!#6;!#1]~[!#6;!#1]", 0),
    126: ("*!@[#8]!@*", 0),
    127: ("*@*!@[#8]", 1),
    130: ("[!#6;!#1]~[!#6;!#1]", 1),
    131: ("[!#6;!#1;!H0]", 1),
    132: ("[#8]~*~[CH2]~*", 0),
    133: ("*@*!@[#7]", 0),
    134: ("[F,Cl,Br,I]", 0),
    136: ("[#8]=*", 1),
    138: ("[!#6;!#1]~[CH2]~*", 1),
    139: ("[O;!H0]", 0),
    140: ("[#8]", 3),
    141: ("[CH3]", 2),
    142: ("[#7]", 1),
    143: ("*@*!@[#8]", 0),
    145: ("*1~*~*~*~*~*~1", 1),
    146: ("[#8]", 2),
    148: ("*~[!#6;!#1](~*)~*", 0),
    149: ("[C;H3,H4]", 1),
    150: ("*!@*@*!@*", 0),
    151: ("[#7;!H0]", 0),
    152: ("[#8]~[#6](~[#6])~[#6]", 0),
    153: ("[!#6;!#1]~[CH2]~*", 0),
    154: ("[#6]=[#8]", 0),
    155: ("*!@[CH2]!@*", 0),
    156: ("[#7]~*(~*)~*", 0),
    157: ("[#6]-[#8]", 0),
    158: ("[#6]-[#7]", 0),
    159: ("[#8]", 1),
    160: ("[C;H3,H4]", 0),
    161: ("[#7]", 0),
    163: ("*1~*~*~*~*~*~1", 0),
    164: ("[#8]", 0),
}

atom_groups = {
    "!#6;!#1": ["#2", "#3", "#4", "#5", "#7", "#8", "#10"],
    "!#6;!#1;!H0": ["#2", "#3", "#4", "#5", "#7", "#8", "#10"],
    "*": ["#6"],
    "~": ["-"],
    "R": ["#6"],
}


# =====================================================================
# OGB feature helpers + mol2graph — verbatim from Mol-LLM
# =====================================================================
def safe_index(l, e):
    try:
        return l.index(e)
    except ValueError:
        return len(l) - 1


def atom_to_feature_vector(atom):
    return [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        safe_index(allowable_features["possible_chirality_list"], str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(allowable_features["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features["possible_hybridization_list"], str(atom.GetHybridization())),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]


def bond_to_feature_vector(bond):
    return [
        safe_index(allowable_features["possible_bond_type_list"], str(bond.GetBondType())),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]


def mol2graph(mol):
    atom_features = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    x = np.array(atom_features, dtype=np.int64)
    num_bond_features = 3
    if mol.GetNumBonds() > 0:
        edges_list, edge_features = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ef = bond_to_feature_vector(bond)
            edges_list.append((i, j)); edge_features.append(ef)
            edges_list.append((j, i)); edge_features.append(ef)
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
    return {
        "edge_index": edge_index,
        "edge_feat": edge_attr,
        "node_feat": x,
        "num_nodes": len(x),
    }


# =====================================================================
# SMARTS specialization + replacement attach — verbatim from Mol-LLM
# =====================================================================
def make_specific_smarts(smarts):
    """Resolve wildcards/alternatives in a SMARTS pattern to a concrete one."""
    pattern = r"(\[\$\(.*?\)\]|[~=#:@\-\+*]|1|\[[^]]+\])"
    parts = re.findall(pattern, smarts)
    result = []
    for part in parts:
        if part.startswith("[$(") and part.endswith(")]"):
            nested = part[2:-1].split("),$(")
            result.append(make_specific_smarts(random.choice(nested)))
        elif part.startswith("[") and part.endswith("]"):
            group = part.strip("[]")
            if group in atom_groups:
                result.append(f"[{random.choice(atom_groups[group])}]")
            elif ";" in group and "," in group:
                base, sub = group.split(";")
                result.append(f"[{base}{random.choice(sub.split(','))}]")
            elif "," in group:
                result.append(f"[{random.choice(group.split(','))}]")
            else:
                result.append(part)
        elif part == "*":
            result.append(f"[{random.choice(atom_groups['*'])}]")
        elif part in ("~", "@"):
            result.append(random.choice(atom_groups["~"]))
        else:
            result.append(part)
    return "".join(result)


def add_replacement_mol(editable_mol, replacement_mol, attach_idx):
    """Add all atoms+bonds of replacement_mol into editable_mol and bond
    replacement[0] to editable_mol[attach_idx] with a single bond.

    Mutates editable_mol in place and returns it (verbatim from Mol-LLM
    augment_dataset.py L460-L491). The caller is responsible for working
    on a Chem.RWMol snapshot if rollback semantics are desired.
    """
    atom_map = {}
    for atom in replacement_mol.GetAtoms():
        atom_map[atom.GetIdx()] = editable_mol.AddAtom(atom)
    for bond in replacement_mol.GetBonds():
        editable_mol.AddBond(
            atom_map[bond.GetBeginAtomIdx()],
            atom_map[bond.GetEndAtomIdx()],
            bond.GetBondType(),
        )
    editable_mol.AddBond(attach_idx, atom_map[0], Chem.BondType.SINGLE)
    editable_mol.UpdatePropertyCache(strict=False)
    return editable_mol


# =====================================================================
# Atomwise extract_and_modify
# =====================================================================
DEFAULT_REPLACE_RATIO = 0.3
DEFAULT_MIN_ATOMS = 4
DEFAULT_MAX_RETRY = 5
DEFAULT_TANIMOTO_FLOOR = 0.05
DEFAULT_TANIMOTO_CEIL = 0.97
CCC_FALLBACK_SMILES = "CCC"

# Atomic numbers used by the atom-swap fallback (last-resort negative).
# Restricted to common organic-chemistry elements that occur in chebi.
_ATOM_SWAP_CANDIDATES = (6, 7, 8, 16)  # C, N, O, S


def _log(msg, verbose):
    if verbose:
        print(msg, file=sys.stderr)


def _min_preserved_atoms(original_n: int) -> int:
    """Lower bound for atom count during the remove loop.

    Larger molecules can tolerate aggressive removal; tiny molecules
    must be protected so that one big-substructure match doesn't drain
    them to zero atoms (the root cause of `empty_graph` CCC fallback).
    """
    return max(2, original_n // 4)


def _atom_swap_fallback(orig_mol, original_smiles, replace_ratio, verbose):
    """Last-resort fallback: clone the original molecule and swap one
    atom's element to produce a chemically plausible negative.

    Returns a result dict shaped like `extract_and_modify_atomwise`'s
    success path, or None if every swap candidate fails sanitization.
    `fallback_reason` is set to `"atom_swap_fallback"` so downstream code
    can tell this came from the swap path (rather than a clean run).
    """
    if orig_mol is None or orig_mol.GetNumAtoms() == 0:
        return None
    indices = list(range(orig_mol.GetNumAtoms()))
    random.shuffle(indices)
    for idx in indices:
        atom = orig_mol.GetAtomWithIdx(idx)
        current_z = atom.GetAtomicNum()
        candidates = [z for z in _ATOM_SWAP_CANDIDATES if z != current_z]
        random.shuffle(candidates)
        for new_z in candidates:
            snapshot = Chem.RWMol(orig_mol)
            snapshot.GetAtomWithIdx(idx).SetAtomicNum(new_z)
            try:
                Chem.SanitizeMol(snapshot)
            except Exception:
                continue
            try:
                modified_smiles = Chem.MolToSmiles(snapshot)
            except Exception:
                modified_smiles = ""
            modified_graph = mol2graph(snapshot)
            if modified_graph["num_nodes"] == 0:
                continue
            _log(f"[atomwise] atom_swap_fallback idx={idx} new_z={new_z} "
                 f"smiles={original_smiles}", verbose)
            return {
                "original_smiles": original_smiles,
                "num_of_key_substructures": 0,
                "n_removed": 0,
                "n_added": 0,
                "atom_change_ratio": 0.0,
                "target_change_ratio": float(replace_ratio),
                "fallback_reason": "atom_swap_fallback",
                "substructures_removed": None,
                "substructures_added": None,
                "removed_graph": mol2graph(orig_mol),
                "modified_smiles": modified_smiles,
                "modified_graph": modified_graph,
            }
    return None


def _ccc_payload(smiles, n_active, replace_ratio, reason, verbose):
    _log(f"[atomwise] fallback={reason} smiles={smiles}", verbose)
    return {
        "original_smiles": smiles,
        "num_of_key_substructures": n_active,
        "n_removed": 0,
        "n_added": 0,
        "atom_change_ratio": 0.0,
        "target_change_ratio": float(replace_ratio),
        "fallback_reason": reason,
        "substructures_removed": None,
        "substructures_added": None,
        "removed_graph": None,
        "modified_smiles": CCC_FALLBACK_SMILES,
        "modified_graph": mol2graph(Chem.MolFromSmiles(CCC_FALLBACK_SMILES)),
    }


def extract_and_modify_atomwise(
    selfies: str,
    replace_ratio: float = DEFAULT_REPLACE_RATIO,
    *,
    verbose: bool = False,
) -> dict:
    """Build a corrupted Mol via MACCS-key remove + whole-substructure add.

    Mirrors the return contract of Mol-LLM's `extract_and_modify` but:
      - The add loop attaches the *entire* replacement substructure
        (Mol-LLM intent — see docs/GRAPH_AUG_ATOMWISE.md §3).
      - Each attach attempt uses a Chem.RWMol snapshot for rollback.
      - Each accepted attach passes Chem.SanitizeMol.
      - Returns extra keys: n_removed, n_added, atom_change_ratio,
        target_change_ratio, fallback_reason.
    """
    smiles = sf.decoder(selfies)
    if not smiles:
        raise ValueError(f"Invalid SELFIES: {selfies!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to parse SMILES: {smiles!r}")
    original_n = mol.GetNumAtoms()

    # --- collect active MACCS substructures + unused SMARTS pool ---
    maccs = MACCSkeys.GenMACCSKeys(mol)
    active = []
    for key, (smarts, _) in smartsPatts.items():
        if not maccs.GetBit(key):
            continue
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        active.append({
            "key": key,
            "smarts": smarts,
            "substructures": [Chem.MolFragmentToSmiles(mol, m) for m in matches],
            "atom_indices": [list(m) for m in matches],
        })
    unused = []
    for key, (smarts, _) in smartsPatts.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and not mol.HasSubstructMatch(patt):
            unused.append(smarts)

    n = len(active)
    num_to_modify = max(int(replace_ratio * n), 1)
    if num_to_modify == 0 or not active or not unused:
        return _ccc_payload(smiles, n, replace_ratio, "no_active_or_unused", verbose)

    to_remove = random.sample(active, num_to_modify)
    to_add = random.sample(unused, num_to_modify)

    # --- remove loop (with minimum-preservation guard) ---
    # Guard prevents draining the molecule to 0 atoms (the root cause of
    # the `empty_graph` CCC fallback). Substructure matches that would
    # drop the count below `min_preserved` are skipped.
    min_preserved = _min_preserved_atoms(original_n)
    editable_mol = Chem.RWMol(mol)
    n_before_remove = editable_mol.GetNumAtoms()
    for sub in to_remove:
        patt = Chem.MolFromSmarts(sub["smarts"])
        if patt is None:
            continue
        matches = editable_mol.GetSubstructMatches(patt)
        if not matches:
            continue
        for _ in range(len(matches)):
            matches = editable_mol.GetSubstructMatches(patt)
            if not matches:
                break
            match = matches[0]
            if editable_mol.GetNumAtoms() - len(match) < min_preserved:
                break  # would drop below preservation threshold
            try:
                rw = Chem.RWMol(editable_mol)
                rw.BeginBatchEdit()
                for aid in match:
                    rw.RemoveAtom(aid)
                rw.CommitBatchEdit()
                Chem.SanitizeMol(rw)
                editable_mol = rw
            except Exception:
                continue
    n_removed = max(0, n_before_remove - editable_mol.GetNumAtoms())
    editable_mol = Chem.RWMol(editable_mol)
    removed_graph = mol2graph(editable_mol)

    # --- add loop (the new behavior) ---
    # Restores the upstream-intended `add_replacement_mol` call with three
    # safety nets vs. simply un-commenting it:
    #   (1) Chem.RWMol(editable_mol) snapshot — real deep copy for rollback
    #   (2) Chem.SanitizeMol — reject invalid-valence attaches
    #   (3) shuffled attach indices — try valence-loose sites first
    n_before_add = editable_mol.GetNumAtoms()
    for smarts in to_add:
        new_smarts = make_specific_smarts(smarts)
        replacement = Chem.MolFromSmarts(new_smarts)
        if replacement is None or replacement.GetNumAtoms() == 0:
            continue
        replacement = Chem.RWMol(replacement)

        if editable_mol.GetNumAtoms() == 0:
            continue

        candidates = list(range(editable_mol.GetNumAtoms()))
        random.shuffle(candidates)
        for attach_idx in candidates:
            snapshot = Chem.RWMol(editable_mol)
            try:
                add_replacement_mol(snapshot, replacement, attach_idx)
                Chem.SanitizeMol(snapshot)
                editable_mol = snapshot
                break
            except Exception as e:
                _log(
                    f"[atomwise] attach fail idx={attach_idx} "
                    f"smarts={new_smarts}: {type(e).__name__}: {e}",
                    verbose,
                )
                continue
    n_added = max(0, editable_mol.GetNumAtoms() - n_before_add)

    modified_graph = mol2graph(editable_mol)
    if modified_graph["num_nodes"] == 0:
        # Last-resort before CCC: try atom-swap (same final position as
        # Mol-LLM's CCC branch — only fires when the molecule is truly
        # empty after the preservation guard couldn't save it).
        swap_result = _atom_swap_fallback(mol, smiles, replace_ratio, verbose)
        if swap_result is not None:
            return swap_result
        return _ccc_payload(smiles, n, replace_ratio, "empty_graph", verbose)

    try:
        modified_smiles = Chem.MolToSmiles(editable_mol)
    except Exception:
        modified_smiles = ""

    atom_change_ratio = (n_removed + n_added) / max(original_n, 1)

    return {
        "original_smiles": smiles,
        "num_of_key_substructures": n,
        "n_removed": int(n_removed),
        "n_added": int(n_added),
        "atom_change_ratio": float(atom_change_ratio),
        "target_change_ratio": float(replace_ratio),
        "fallback_reason": None,
        "substructures_removed": to_remove,
        "substructures_added": to_add,
        "removed_graph": removed_graph,
        "modified_smiles": modified_smiles,
        "modified_graph": modified_graph,
    }


# =====================================================================
# High-level adapter (V-MolPO collator-friendly output + guards)
# =====================================================================
def selfies_from_input_mol_string(input_mol_string: str) -> str:
    return (
        input_mol_string.replace("<SELFIES>", "")
        .replace("</SELFIES>", "")
        .strip()
    )


def smiles_to_selfies(smiles: str) -> str | None:
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def _seed_globals(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _maccs_tanimoto(smiles_a: str, smiles_b: str) -> float | None:
    a = Chem.MolFromSmiles(smiles_a)
    b = Chem.MolFromSmiles(smiles_b)
    if a is None or b is None:
        return None
    try:
        return DataStructs.TanimotoSimilarity(
            MACCSkeys.GenMACCSKeys(a),
            MACCSkeys.GenMACCSKeys(b),
        )
    except Exception:
        return None


def _try_atomwise(
    selfies: str,
    replace_ratio: float,
    seed: int,
    *,
    min_atoms: int,
    tanimoto_floor: float,
    tanimoto_ceil: float,
    verbose: bool,
):
    _seed_globals(seed)
    try:
        out = extract_and_modify_atomwise(
            selfies, replace_ratio=replace_ratio, verbose=verbose,
        )
    except Exception as e:
        return f"extract_exception:{type(e).__name__}"
    if out.get("fallback_reason") is not None:
        return f"fallback:{out['fallback_reason']}"
    if out.get("modified_smiles") == CCC_FALLBACK_SMILES:
        return "ccc_fallback"

    g = out.get("modified_graph") or {}
    if g.get("num_nodes", 0) < min_atoms:
        return "below_min_atoms"
    node_feat = g.get("node_feat")
    edge_index = g.get("edge_index")
    edge_feat = g.get("edge_feat")
    if node_feat is None or edge_index is None or edge_feat is None:
        return "missing_graph_fields"

    tani: float | None = None
    if out.get("modified_smiles"):
        tani = _maccs_tanimoto(out["original_smiles"], out["modified_smiles"])
        if tani is not None and not (tanimoto_floor <= tani <= tanimoto_ceil):
            return f"tanimoto_out_of_band:{tani:.3f}"

    return {
        "x": node_feat.tolist(),
        "edge_index": edge_index.tolist(),
        "edge_attr": edge_feat.tolist(),
        "atom_change_ratio": float(out["atom_change_ratio"]),
        "n_removed": int(out["n_removed"]),
        "n_added": int(out["n_added"]),
        "tanimoto": float(tani) if tani is not None else -1.0,
        "modified_smiles": out["modified_smiles"] or "",
    }


def augment_molecule_graph_atomwise(
    selfies_str: str,
    *,
    replace_ratio: float = DEFAULT_REPLACE_RATIO,
    min_atoms: int = DEFAULT_MIN_ATOMS,
    max_retry: int = DEFAULT_MAX_RETRY,
    tanimoto_floor: float = DEFAULT_TANIMOTO_FLOOR,
    tanimoto_ceil: float = DEFAULT_TANIMOTO_CEIL,
    seed: int = 42,
    return_reason: bool = False,
    verbose: bool = False,
):
    """Build a paper-faithful rejected graph with whole-substructure addition.

    Returns dict {x, edge_index, edge_attr, atom_change_ratio, n_removed,
    n_added, tanimoto, modified_smiles} on success, or None on failure.
    With return_reason=True, returns (result_or_None, reason_string).
    """
    def _ret(val, reason="ok"):
        return (val, reason) if return_reason else val

    if not isinstance(selfies_str, str) or not selfies_str.strip():
        return _ret(None, "empty_input")
    if selfies_str == "<None>":
        return _ret(None, "none_marker")

    last_reason = None
    for attempt in range(max_retry):
        result = _try_atomwise(
            selfies_str,
            replace_ratio=replace_ratio,
            seed=seed + attempt * 7919,
            min_atoms=min_atoms,
            tanimoto_floor=tanimoto_floor,
            tanimoto_ceil=tanimoto_ceil,
            verbose=verbose,
        )
        if isinstance(result, dict):
            return _ret(result, "ok")
        last_reason = result
    return _ret(None, last_reason or "max_retry_exhausted")


# =====================================================================
# FTS-targeted variant (0.7 algorithm) — best-of-N attempts
# =====================================================================
# The base `extract_and_modify_atomwise` (batch mode, "0.55 algorithm")
# produces FTS ≈ 0.55 mean on chebi-20-mol2text because even one
# substructure modification cascades across many MACCS bits. The
# `_fts_targeted` variant tries up to `max_attempts` different random
# states and keeps the attempt whose MACCS Tanimoto is closest to a
# target value (default 0.7 — Mol-LLM §B.4 implied result metric).

DEFAULT_TARGET_FTS = 0.7
DEFAULT_TARGET_FTS_TOLERANCE = 0.10  # accept anything in [0.60, 0.80]
DEFAULT_FTS_MAX_ATTEMPTS = 10
DEFAULT_FTS_TARGET_REPLACE_RATIO = 0.05  # smaller ratio = finer targeting


def _maccs_tanimoto_mol(orig_mol, mod_mol) -> float | None:
    if orig_mol is None or mod_mol is None:
        return None
    try:
        return DataStructs.TanimotoSimilarity(
            MACCSkeys.GenMACCSKeys(orig_mol),
            MACCSkeys.GenMACCSKeys(mod_mol),
        )
    except Exception:
        return None


def extract_and_modify_fts_targeted(
    selfies: str,
    target_fts: float = DEFAULT_TARGET_FTS,
    target_fts_tolerance: float = DEFAULT_TARGET_FTS_TOLERANCE,
    replace_ratio: float = DEFAULT_FTS_TARGET_REPLACE_RATIO,
    max_attempts: int = DEFAULT_FTS_MAX_ATTEMPTS,
    *,
    verbose: bool = False,
) -> dict:
    """Best-of-N wrapper around `extract_and_modify_atomwise` targeting a
    specific MACCS FTS value (Mol-LLM §B.4 implied evaluation metric).

    Calls the base function up to `max_attempts` times with diverging
    random state (consumed by `random.choice`/`random.sample` internally).
    Returns the first attempt whose MACCS Tanimoto falls within
    `[target_fts - tol, target_fts + tol]`. If none lands in the band,
    returns the attempt whose FTS is closest to `target_fts`.

    Result dict includes extra keys: `maccs_fts`, `n_fts_attempts`,
    `fts_in_band`.

    Use this when you need MACCS FTS ≈ target (e.g. 0.7). Use the base
    `extract_and_modify_atomwise` when you don't care about FTS targeting
    (faster, single attempt, but FTS averages ~0.55 on drug-like mols).
    """
    smiles = sf.decoder(selfies)
    if not smiles:
        raise ValueError(f"Invalid SELFIES: {selfies!r}")
    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        raise ValueError(f"Unable to parse SMILES: {smiles!r}")

    band_low = target_fts - target_fts_tolerance
    band_high = target_fts + target_fts_tolerance

    best_out: dict | None = None
    best_fts: float | None = None
    best_dist = float("inf")
    n_fallback = 0

    for attempt in range(max_attempts):
        # No explicit seeding here — caller-set global random state
        # diverges across attempts because the base function consumes
        # randomness internally (random.sample / random.choice).
        out = extract_and_modify_atomwise(
            selfies, replace_ratio=replace_ratio, verbose=verbose,
        )
        if out.get("fallback_reason") is not None:
            n_fallback += 1
            continue

        mod_mol = Chem.MolFromSmiles(out["modified_smiles"])
        fts = _maccs_tanimoto_mol(orig_mol, mod_mol)
        if fts is None:
            n_fallback += 1
            continue

        out["maccs_fts"] = float(fts)
        out["n_fts_attempts"] = attempt + 1
        out["target_fts"] = float(target_fts)

        if band_low <= fts <= band_high:
            out["fts_in_band"] = True
            return out

        dist = abs(fts - target_fts)
        if dist < best_dist:
            best_dist = dist
            best_fts = fts
            best_out = out

    if best_out is None:
        # Every attempt fell into CCC fallback or unparseable SMILES.
        # Try atom-swap as a last-resort negative before declaring CCC.
        _log(f"[fts_targeted] all {max_attempts} attempts failed for {smiles}", verbose)
        swap_result = _atom_swap_fallback(orig_mol, smiles, replace_ratio, verbose)
        if swap_result is not None:
            return swap_result
        return _ccc_payload(smiles, 0, replace_ratio, "all_fts_attempts_failed", verbose)

    best_out["fts_in_band"] = False
    best_out["n_fts_attempts"] = max_attempts
    if verbose:
        _log(f"[fts_targeted] no attempt in band, returning closest "
             f"(fts={best_fts:.3f}, target={target_fts}) for {smiles}", verbose)
    return best_out


# =====================================================================
# Dataset-level map function — produces {i}-th_rejected_* keys
# =====================================================================
# Mirrors `/opt/mol-llm_official/augment_dataset.py:851-923`
# `map_by_substructure_replacement` and Old_MolDA's `:647-652` keying
# convention. Stage 3 collators consume `{i}-th_rejected_*` and pick one
# `i` per epoch via `reject_cardinal = current_epoch`.

# Tasks where the row's input is text (no real molecule to corrupt). For
# these we generate dummy graphs from `[C][C][C]` exactly like Mol-LLM
# upstream (`augment_dataset.py:890-901`, `TEXT2MOL_BENCHMARKS`). DPO loss
# still applies — the dummy serves as noise/control while preference signal
# comes from LLM logits on the text output.
#
# Membership matches Mol-LLM's TEXT2MOL_BENCHMARKS. name_conversion tasks
# (i2s, i2f, s2i) are Stage-1-only and excluded from this build via
# --exclude-tasks at build time, so they don't need to be in this set.
TEXT2MOL_LIKE_TASKS = frozenset({
    "chebi-20-text2mol",
    "smol-molecule_generation",
})

_DUMMY_TEXT2MOL_SELFIES = "[C][C][C]"
_CCC_GRAPH_CACHE: dict | None = None


def _ccc_graph() -> dict:
    """Pre-built CCC fallback graph (matches Old_MolDA :640-664)."""
    global _CCC_GRAPH_CACHE
    if _CCC_GRAPH_CACHE is None:
        _CCC_GRAPH_CACHE = mol2graph(Chem.MolFromSmiles(CCC_FALLBACK_SMILES))
    return _CCC_GRAPH_CACHE


def _graph_to_lists(graph: dict) -> tuple:
    """Convert mol2graph numpy output to nested-list form (HF Datasets safe)."""
    return (
        graph["node_feat"].tolist(),
        graph["edge_index"].tolist(),
        graph["edge_feat"].tolist(),
    )


def _safe_atomwise_graph(
    selfies: str,
    replace_ratio: float,
    verbose: bool,
    *,
    target_fts: float | None = None,
    target_fts_tolerance: float = DEFAULT_TARGET_FTS_TOLERANCE,
    max_attempts: int = DEFAULT_FTS_MAX_ATTEMPTS,
) -> dict:
    """One atomwise call → modified_graph dict. Falls back to CCC on any failure
    or when the result graph has < 2 nodes (Old_MolDA :640-644 guard).

    When `target_fts` is None (default), uses the base batch algorithm
    (≈0.55 FTS on chebi). When set, uses `extract_and_modify_fts_targeted`
    to converge MACCS Tanimoto toward `target_fts`.
    """
    try:
        if target_fts is None:
            out = extract_and_modify_atomwise(
                selfies, replace_ratio=replace_ratio, verbose=verbose,
            )
        else:
            out = extract_and_modify_fts_targeted(
                selfies,
                target_fts=target_fts,
                target_fts_tolerance=target_fts_tolerance,
                replace_ratio=replace_ratio,
                max_attempts=max_attempts,
                verbose=verbose,
            )
    except Exception as e:
        if verbose:
            print(f"[atomwise-map] exception {type(e).__name__}: {e}", file=sys.stderr)
        return _ccc_graph()
    g = out.get("modified_graph") or {}
    if g.get("num_nodes", 0) < 2:
        return _ccc_graph()
    return g


def _strip_selfies_wrapper(s: str) -> str:
    return (
        s.replace("<SELFIES>", "")
        .replace("</SELFIES>", "")
        .replace(" ", "")
    )


def map_by_substructure_replacement_atomwise(
    data_point: dict,
    replace_ratio: float = DEFAULT_REPLACE_RATIO,
    num_rejected_graphs: int = 6,
    selfies_field: str = "input_mol_string_selfies",
    fallback_selfies_field: str = "input_mol_string",
    *,
    target_fts: float | None = None,
    target_fts_tolerance: float = DEFAULT_TARGET_FTS_TOLERANCE,
    max_attempts: int = DEFAULT_FTS_MAX_ATTEMPTS,
    skip_additional_rejected: bool = False,
    verbose: bool = False,
) -> dict:
    """Populate `{i}-th_rejected_*` and `{i}-th_additional_rejected_*` keys
    on a dataset row, producing `num_rejected_graphs` atomwise-corrupted
    variants per sample.

    Two algorithm modes selectable per call:
      - **batch mode** (default, `target_fts=None`): one-shot
        `extract_and_modify_atomwise` per slot. Faster. MACCS FTS averages
        ~0.55 on chebi (substructure cascade pulls it below the 0.7 target).
      - **fts-targeted mode** (`target_fts=0.7`): best-of-N attempts via
        `extract_and_modify_fts_targeted`. Slower (up to `max_attempts` ×
        per-attempt cost) but FTS converges toward the target band.

    Contract mirrors mol-llm_official `map_by_substructure_replacement`
    (augment_dataset.py:851-923) and Old_MolDA's :647-652 key layout.

    Task dispatch (matches upstream):
      - Reaction tasks (selfies contains `|>>|`): two halves corrupted
        independently — first → rejected, second → additional_rejected.
      - Text2mol / name-conversion tasks (`TEXT2MOL_LIKE_TASKS`): both
        slots use a dummy `[C][C][C]` SELFIES.
      - Default (e.g. chebi-20-mol2text): same SELFIES corrupted twice
        per i for `rejected` and `additional_rejected`.

    Args:
        data_point: HuggingFace row dict (mutated in place + returned).
        replace_ratio: forwarded to underlying extract function. Default
            0.3 for batch mode; for fts-targeted mode 0.05 is recommended
            (passed in explicitly if used).
        num_rejected_graphs: number of i values to emit (Old_MolDA: 6).
        selfies_field: row column holding the SELFIES string. MolDA's
            dual-column dataset uses `input_mol_string_selfies`.
        fallback_selfies_field: used when `selfies_field` is missing
            (mol-llm raw format uses `input_mol_string`).
        target_fts: when set, switch to fts-targeted mode. Recommended
            value 0.7 (Mol-LLM §B.4 implied evaluation metric).
        target_fts_tolerance: accept band [target_fts ± tol]. Default 0.10.
        max_attempts: best-of-N retries in fts-targeted mode.
        verbose: stderr diagnostics on per-attempt failures.
    """
    replace_ratio = min(replace_ratio, 1.0)
    task = data_point.get("task", "") or ""

    selfies_raw = data_point.get(selfies_field) or data_point.get(fallback_selfies_field) or ""
    selfies = _strip_selfies_wrapper(selfies_raw)

    is_reaction = "|>>|" in selfies
    is_text2mol = task in TEXT2MOL_LIKE_TASKS

    kw = dict(
        target_fts=target_fts,
        target_fts_tolerance=target_fts_tolerance,
        max_attempts=max_attempts,
    )

    # Reaction tasks ALWAYS need the additional graph (it's the product
    # half of `reactant|>>|product`). For mol2text / text2mol / regression
    # tasks `additional_rejected_*` is just a second independent corruption
    # of the same molecule — the downstream Old_MolDA collator only reads it
    # for reaction tasks (`list_additional_graphs`), so it can be safely
    # skipped elsewhere via `skip_additional_rejected=True`.
    skip_additional = skip_additional_rejected and not is_reaction

    kw = dict(
        target_fts=target_fts,
        target_fts_tolerance=target_fts_tolerance,
        max_attempts=max_attempts,
    )

    for i in range(num_rejected_graphs):
        if is_reaction:
            pair = selfies.split("|>>|")
            rejected = _safe_atomwise_graph(pair[0], replace_ratio, verbose, **kw)
            additional = _safe_atomwise_graph(
                pair[1] if len(pair) > 1 else pair[0],
                replace_ratio, verbose, **kw)
        elif is_text2mol:
            rejected = _safe_atomwise_graph(_DUMMY_TEXT2MOL_SELFIES, replace_ratio, verbose, **kw)
            additional = (
                None if skip_additional
                else _safe_atomwise_graph(_DUMMY_TEXT2MOL_SELFIES, replace_ratio, verbose, **kw)
            )
        else:
            rejected = _safe_atomwise_graph(selfies, replace_ratio, verbose, **kw)
            additional = (
                None if skip_additional
                else _safe_atomwise_graph(selfies, replace_ratio, verbose, **kw)
            )

        r_x, r_ei, r_ea = _graph_to_lists(rejected)
        data_point[f"{i}-th_rejected_x"] = r_x
        data_point[f"{i}-th_rejected_edge_index"] = r_ei
        data_point[f"{i}-th_rejected_edge_attr"] = r_ea

        # Schema-consistency: always emit `additional_rejected_*` keys so that
        # mixed reaction + non-reaction batches don't break HF Arrow writer's
        # schema inference. For non-reaction rows (additional is None due to
        # skip_additional_rejected) we use a CCC dummy graph — collator only
        # reads `additional_*` for reaction tasks, so the dummy is unused.
        if additional is not None:
            a_x, a_ei, a_ea = _graph_to_lists(additional)
        else:
            a_x, a_ei, a_ea = _graph_to_lists(_ccc_graph())
        data_point[f"{i}-th_additional_rejected_x"] = a_x
        data_point[f"{i}-th_additional_rejected_edge_index"] = a_ei
        data_point[f"{i}-th_additional_rejected_edge_attr"] = a_ea

    return data_point
