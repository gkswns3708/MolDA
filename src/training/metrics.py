"""
Task-specific evaluation metrics.

Task categorization synced with benchmark_constants.py (all benchmarks covered).
Calculation logic aligned with Old_MolDA/model/help_funcs.py.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Task categorization (synced with benchmark_constants.py)
# ─────────────────────────────────────────────

CLASSIFICATION_TASKS = {
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
    "tox21",
    "toxcast",
}

REGRESSION_TASKS = {
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
}

REACTION_TASKS = {
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
}

TEXT2MOL_TASKS = {
    "chebi-20-text2mol",
    "smol-molecule_generation",
}

MOL2TEXT_TASKS = {
    "chebi-20-mol2text",
    "smol-molecule_captioning",
}

NAME_CONVERSION_TASKS = {
    "smol-name_conversion-i2s",
    "smol-name_conversion-i2f",
    "smol-name_conversion-s2f",
    "smol-name_conversion-s2i",
}

# Union of all generation-based tasks (everything except classification and name_conversion)
GENERATION_TASKS = REGRESSION_TASKS | REACTION_TASKS | TEXT2MOL_TASKS | MOL2TEXT_TASKS

ALL_TASKS = (CLASSIFICATION_TASKS | REGRESSION_TASKS | REACTION_TASKS
             | TEXT2MOL_TASKS | MOL2TEXT_TASKS | NAME_CONVERSION_TASKS)


def get_task_type(task: str) -> str:
    if task in CLASSIFICATION_TASKS:
        return "classification"
    elif task in REGRESSION_TASKS:
        return "regression"
    elif task in REACTION_TASKS or task in TEXT2MOL_TASKS:
        return "molecule"
    elif task in MOL2TEXT_TASKS:
        return "caption"
    elif task in NAME_CONVERSION_TASKS:
        return "name_conversion"
    else:
        logger.warning(f"Unknown task: {task}, treating as molecule")
        return "molecule"


# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def _parse_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <TAG>...</TAG>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_tag_with_fallback(text: str, tag: str) -> Optional[str]:
    """Extract content between <TAG>...</TAG> with left-side fallback.

    Tries dual-side pattern first (closed tag), then left-side (open-ended).
    Matches Old_MolDA regex strategy.
    """
    dual = re.search(rf"(?<=<{tag}>).*?(?=</{tag}>)", text, re.DOTALL)
    if dual:
        return dual.group()
    left = re.search(rf"(?<=<{tag}>).*", text, re.DOTALL)
    if left:
        return left.group()
    return None


def _parse_float_tag(text: str) -> Optional[float]:
    """Extract float from <FLOAT>...</FLOAT>."""
    content = _parse_tag(text, "FLOAT")
    if content is None:
        return None
    # Remove special number tokens like <|0|>, <|.|>, etc.
    cleaned = re.sub(r"<\|(.)\|>", r"\1", content).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_boolean_tag(text: str) -> Optional[bool]:
    """Extract boolean from <BOOLEAN>...</BOOLEAN>.

    Multi-label case (e.g. "True, True, False"): returns True if any value is True.
    """
    content = _parse_tag(text, "BOOLEAN")
    if content is None:
        return None
    content = content.strip().lower()
    # Multi-label: comma-separated booleans → reduce to any-positive
    if "," in content:
        parts = [p.strip() for p in content.split(",")]
        return any(p == "true" for p in parts)
    if content == "true":
        return True
    elif content == "false":
        return False
    return None


# ─────────────────────────────────────────────
# Classification: likelihood-based eval
# ─────────────────────────────────────────────

def classification_evaluate(probs: "torch.Tensor", label_texts: List[str],
                            task: str) -> Dict[str, float]:
    """Evaluate binary classification using likelihood probabilities.

    Args:
        probs: [N, 2] tensor of [P(False), P(True)]
        label_texts: list of ground truth strings e.g. "<BOOLEAN> True </BOOLEAN>"
        task: task name

    Returns:
        dict with accuracy, f1, roc_auc, failure_rate
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

    gt_labels = []
    valid_indices = []
    failure_idxs = []
    for i, text in enumerate(label_texts):
        parsed = _parse_boolean_tag(text)
        if parsed is not None:
            gt_labels.append(1 if parsed else 0)
            valid_indices.append(i)
        else:
            failure_idxs.append(i)

    failure_rate = 1.0 - len(valid_indices) / len(label_texts) if len(label_texts) > 0 else 0.0

    if not gt_labels:
        return {
            "accuracy": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "roc_auc": float("nan"),
            "failure_rate": failure_rate,
            "_failure_indices": failure_idxs,
        }

    gt = np.array(gt_labels)
    valid_probs = probs[valid_indices]
    prob_true = valid_probs[:, 1].cpu().numpy()
    preds = valid_probs.argmax(dim=-1).cpu().numpy()

    acc = float(accuracy_score(gt, preds))
    f1 = float(f1_score(gt, preds))
    prec = float(precision_score(gt, preds))
    rec = float(recall_score(gt, preds))

    try:
        roc_auc = float(roc_auc_score(gt, prob_true))
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
        "failure_rate": failure_rate,
        "_failure_indices": failure_idxs,
    }


# ─────────────────────────────────────────────
# Regression: generation + float parse
# ─────────────────────────────────────────────

def regression_evaluate(pred_texts: List[str], label_texts: List[str],
                        task: str) -> Dict[str, float]:
    """Evaluate regression by parsing <FLOAT> tags and computing MAE/RMSE.

    Matches Old_MolDA regression_evaluate:
    - Label parsing is expected to always succeed.
    - Prediction must contain <|.|> token for proper magnitude.
    """
    preds, gts = [], []
    failure_idxs = []

    for i in range(len(pred_texts)):
        # Label parsing (expected to always succeed for well-formed data)
        label_content = re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", label_texts[i])
        if label_content is None:
            failure_idxs.append(i)
            continue
        label_str = label_content.group().replace(" ", "")
        label_str = label_str.replace("<|", "").replace("|>", "")
        label_val = float(label_str)

        # Prediction parsing with <|.|> validation
        try:
            assert (
                "<|.|>" in pred_texts[i]
            ), f"Prediction should include <|.|> token for proper magnitude of order"
            pred_content = re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", pred_texts[i])
            assert pred_content is not None
            pred_str = pred_content.group().replace(" ", "")
            pred_str = pred_str.replace("<|", "").replace("|>", "")
            pred_val = float(pred_str)

            gts.append(label_val)
            preds.append(pred_val)
        except Exception:
            failure_idxs.append(i)

    n_total = len(pred_texts)
    n_valid = len(preds)

    if n_valid == 0:
        return {"mae": float("inf"), "mse": float("inf"), "rmse": float("inf"),
                "failure_rate": len(failure_idxs) / n_total if n_total > 0 else 0.0,
                "_failure_indices": failure_idxs}

    preds = np.array(preds)
    gts = np.array(gts)

    mae = float(np.mean(np.abs(gts - preds)))
    mse = float(np.mean((gts - preds) ** 2))
    rmse = float(np.mean((gts - preds) ** 2) ** 0.5)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "failure_rate": len(failure_idxs) / n_total if n_total > 0 else 0.0,
        "_failure_indices": failure_idxs,
    }


# ─────────────────────────────────────────────
# Molecule: generation + SELFIES/SMILES parse
# ─────────────────────────────────────────────

def molecule_evaluate(pred_texts: List[str], label_texts: List[str],
                      task: str, tokenizer=None) -> Dict[str, float]:
    """Evaluate molecule generation (reaction, text2mol) by parsing SELFIES or SMILES.

    Auto-detects tag format per sample:
    - <SELFIES> tag → SELFIES mode (sf.decoder → SMILES → RDKit)
    - <SMILES> tag → SMILES mode (direct RDKit processing, no selfies dependency)

    Calculation logic matches Old_MolDA molecule_evaluate:
    - Spaces removed before parsing.
    - Dual-side + left-side fallback regex.
    - Residual tag cleanup on predictions.
    - Exact match via InChI comparison.
    - Levenshtein: raw edit distance on canonical SMILES.
    - Validity = 1 - failure_count / total_count.
    - BLEU always computed with tokenizer.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys, AllChem, DataStructs
        from rdkit.Chem.inchi import MolToInchi
    except ImportError:
        logger.warning("rdkit not installed, returning empty metrics")
        return {"validity_ratio": 0.0, "exact_match_ratio": 0.0,
                "MACCS_FTS": 0.0, "RDK_FTS": 0.0, "morgan_FTS": 0.0,
                "bleu_smiles": 0.0, "bleu_selfies": 0.0,
                "levenshtein_score": 0.0, "failure_rate": 1.0}

    sf = None
    try:
        import selfies as sf
    except ImportError:
        pass  # SMILES 모드에서는 selfies 없어도 동작

    try:
        from Levenshtein import distance as lev_distance
    except ImportError:
        lev_distance = None

    n_total = len(pred_texts)
    failure_idxs = []
    exact_matches = []
    levs = []
    MACCS_sims = []
    RDK_sims = []
    morgan_sims = []
    ref_selfies_list, pred_selfies_list = [], []
    ref_smiles_list, pred_smiles_list = [], []

    for i in range(n_total):
        target = label_texts[i].replace(" ", "")
        prediction = pred_texts[i].replace(" ", "")

        prediction_mol = None
        use_selfies = False
        target_canonical_selfies = None
        prediction_canonical_selfies = None

        try:
            # Auto-detect: SELFIES 태그 먼저 시도, 없으면 SMILES 태그
            target_selfies_raw = _parse_tag_with_fallback(target, "SELFIES")

            if target_selfies_raw is not None:
                # === SELFIES 모드 ===
                use_selfies = True
                assert sf is not None, "selfies package required for SELFIES mode"
                target_smiles = sf.decoder(target_selfies_raw)
                target_mol = Chem.MolFromSmiles(target_smiles)
                target_canonical_smiles = Chem.CanonSmiles(target_smiles)
                target_canonical_selfies = sf.encoder(target_canonical_smiles)

                # Parse prediction SELFIES
                prediction_selfies = _parse_tag_with_fallback(prediction, "SELFIES")
                assert prediction_selfies is not None, "Prediction SELFIES tag not found"

                # Cleanup residual tags
                prediction_selfies = prediction_selfies.split("<SELFIES>")[-1]
                prediction_selfies = prediction_selfies.split("</SELFIES>")[0]
                assert (
                    "<SELFIES>" not in prediction_selfies
                    and "</SELFIES>" not in prediction_selfies
                )

                prediction_smiles = sf.decoder(prediction_selfies)
                prediction_mol = Chem.MolFromSmiles(prediction_smiles)
                prediction_canonical_smiles = Chem.CanonSmiles(prediction_smiles)
                prediction_canonical_selfies = sf.encoder(prediction_canonical_smiles)
            else:
                # === SMILES 모드 ===
                target_smiles_raw = _parse_tag_with_fallback(target, "SMILES")
                assert target_smiles_raw is not None, "No SELFIES or SMILES tag found in target"

                # Cleanup residual tags (prediction)
                target_smiles = target_smiles_raw.split("<SMILES>")[-1].split("</SMILES>")[0]
                target_mol = Chem.MolFromSmiles(target_smiles)
                target_canonical_smiles = Chem.CanonSmiles(target_smiles)

                prediction_smiles_raw = _parse_tag_with_fallback(prediction, "SMILES")
                assert prediction_smiles_raw is not None, "Prediction SMILES tag not found"

                prediction_smiles = prediction_smiles_raw.split("<SMILES>")[-1].split("</SMILES>")[0]
                prediction_mol = Chem.MolFromSmiles(prediction_smiles)
                prediction_canonical_smiles = Chem.CanonSmiles(prediction_smiles)

            # Exact match via InChI (공통)
            exact_matches.append(
                MolToInchi(target_mol) == MolToInchi(prediction_mol)
            )
        except Exception:
            failure_idxs.append(i)
            prediction_mol = None

        if prediction_mol is not None:
            # Levenshtein on canonical SMILES (raw distance)
            if lev_distance is not None:
                levs.append(lev_distance(target_canonical_smiles, prediction_canonical_smiles))
            else:
                levs.append(_levenshtein_distance(target_canonical_smiles, prediction_canonical_smiles))

            # BLEU tokenization (always when tokenizer provided)
            if tokenizer is not None:
                pred_smiles_list.append(tokenizer.tokenize(prediction_canonical_smiles))
                ref_smiles_list.append([tokenizer.tokenize(target_canonical_smiles)])
                # SELFIES BLEU: SELFIES 모드에서만 계산
                if use_selfies and target_canonical_selfies and prediction_canonical_selfies:
                    pred_selfies_list.append(tokenizer.tokenize(prediction_canonical_selfies))
                    ref_selfies_list.append([tokenizer.tokenize(target_canonical_selfies)])

            # Fingerprint Tanimoto Similarities
            MACCS_sims.append(
                DataStructs.FingerprintSimilarity(
                    MACCSkeys.GenMACCSKeys(target_mol),
                    MACCSkeys.GenMACCSKeys(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            RDK_sims.append(
                DataStructs.FingerprintSimilarity(
                    Chem.RDKFingerprint(target_mol),
                    Chem.RDKFingerprint(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            morgan_sims.append(
                DataStructs.TanimotoSimilarity(
                    AllChem.GetMorganFingerprint(target_mol, 2),
                    AllChem.GetMorganFingerprint(prediction_mol, 2),
                )
            )

    # Validity = 1 - failure_count / total (matches Old_MolDA)
    validity_ratio = 1 - len(failure_idxs) / n_total if n_total > 0 else 0.0

    # BLEU scores
    bleu_smiles, bleu_selfies = 0.0, 0.0
    if pred_smiles_list:
        try:
            from nltk.translate.bleu_score import corpus_bleu
            bleu_smiles = corpus_bleu(
                ref_smiles_list, pred_smiles_list,
                weights=(0.25, 0.25, 0.25, 0.25),
            ) * 100
        except ImportError:
            pass
    if pred_selfies_list:
        try:
            from nltk.translate.bleu_score import corpus_bleu
            bleu_selfies = corpus_bleu(
                ref_selfies_list, pred_selfies_list,
                weights=(0.25, 0.25, 0.25, 0.25),
            ) * 100
        except ImportError:
            pass

    return {
        "validity_ratio": validity_ratio,
        "MACCS_FTS": float(np.mean(MACCS_sims)) if MACCS_sims else 0.0,
        "RDK_FTS": float(np.mean(RDK_sims)) if RDK_sims else 0.0,
        "morgan_FTS": float(np.mean(morgan_sims)) if morgan_sims else 0.0,
        "exact_match_ratio": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "levenshtein_score": float(np.mean(levs)) if levs else 0.0,
        "bleu_smiles": float(bleu_smiles),
        "bleu_selfies": float(bleu_selfies),
        "failure_rate": len(failure_idxs) / n_total if n_total > 0 else 0.0,
        "_failure_indices": failure_idxs,
    }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Fallback Levenshtein distance (when python-Levenshtein not installed)."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ─────────────────────────────────────────────
# Caption: generation + text parse
# ─────────────────────────────────────────────

def caption_evaluate(pred_texts: List[str], label_texts: List[str],
                     task: str, tokenizer=None,
                     meteor_tokenizers: Optional[List[str]] = None) -> Dict[str, float]:
    """Evaluate text generation (mol2text, captioning) with BLEU/ROUGE/METEOR.

    Matches Old_MolDA caption_evaluate:
    - Auto-detects tag type from target content.
    - Dual-side + left-side fallback regex.
    - BLEU: word_tokenize(lower) or tokenizer, no smoothing, ×100.
    - METEOR: word_tokenize(lower) for wordnet, tokenizer for llada, ×100.
    - ROUGE: no stemmer, score(hyp, ref) order, ×100.

    Args:
        pred_texts: model-generated strings
        label_texts: ground truth strings
        task: task name
        tokenizer: HF tokenizer for LLaDA-mode METEOR/BLEU
        meteor_tokenizers: list of ["wordnet", "llada"] (default: ["wordnet", "llada"])
    """
    if meteor_tokenizers is None:
        meteor_tokenizers = ["wordnet", "llada"]

    use_wordnet = "wordnet" in meteor_tokenizers
    use_llada = "llada" in meteor_tokenizers

    # Compiled regex patterns for tag auto-detection (matches Old_MolDA)
    patterns = {
        "DESCRIPTION": {
            "dual_side": re.compile(r"(?<=<DESCRIPTION>).*?(?=</DESCRIPTION>)"),
            "left_side": re.compile(r"(?<=<DESCRIPTION>).*"),
        },
        "IUPAC": {
            "dual_side": re.compile(r"(?<=<IUPAC>).*?(?=</IUPAC>)"),
            "left_side": re.compile(r"(?<=<IUPAC>).*"),
        },
        "MOLFORMULA": {
            "dual_side": re.compile(r"(?<=<MOLFORMULA>).*?(?=</MOLFORMULA>)"),
            "left_side": re.compile(r"(?<=<MOLFORMULA>).*"),
        },
    }

    # Tokenization lists
    references_wordnet, hypotheses_wordnet = [], []
    references_llada, hypotheses_llada = [], []
    ref_sentences, hyp_sentences = [], []
    failure_idxs = []

    for i in range(len(label_texts)):
        target = label_texts[i]
        prediction = pred_texts[i]

        # Auto-detect tag from target content (matches Old_MolDA)
        pattern = None
        for key, matching_pattern in patterns.items():
            if matching_pattern["left_side"].search(target):
                pattern = matching_pattern
                break
        if pattern is None:
            failure_idxs.append(i)
            continue

        # Parse reference
        if pattern["dual_side"].search(target):
            ref = pattern["dual_side"].search(target).group()
        else:
            ref = pattern["left_side"].search(target).group()
        ref_sentences.append(ref)

        # Tokenize reference
        if use_wordnet:
            try:
                from nltk.tokenize import word_tokenize
                ref_tokens_wordnet = word_tokenize(ref.lower())
            except ImportError:
                ref_tokens_wordnet = ref.lower().split()
        if use_llada and tokenizer is not None:
            ref_tokens_llada = tokenizer.tokenize(ref)

        # Parse prediction
        try:
            if pattern["dual_side"].search(prediction):
                pred = pattern["dual_side"].search(prediction).group()
            else:
                pred = pattern["left_side"].search(prediction).group()
            hyp_sentences.append(pred)

            # Tokenize prediction
            if use_wordnet:
                try:
                    from nltk.tokenize import word_tokenize
                    pred_tokens_wordnet = word_tokenize(pred.lower())
                except ImportError:
                    pred_tokens_wordnet = pred.lower().split()
                references_wordnet.append([ref_tokens_wordnet])
                hypotheses_wordnet.append(pred_tokens_wordnet)
            if use_llada and tokenizer is not None:
                pred_tokens_llada = tokenizer.tokenize(pred)
                references_llada.append([ref_tokens_llada])
                hypotheses_llada.append(pred_tokens_llada)
        except Exception:
            failure_idxs.append(i)

    has_valid_samples = len(hyp_sentences) > 0

    if has_valid_samples:
        # BLEU: word_tokenize(lower) based, no smoothing, ×100 (matches Old_MolDA)
        try:
            from nltk.translate.bleu_score import corpus_bleu
            if use_wordnet and references_wordnet:
                bleu2 = corpus_bleu(references_wordnet, hypotheses_wordnet, weights=(0.5, 0.5)) * 100
                bleu4 = corpus_bleu(references_wordnet, hypotheses_wordnet, weights=(0.25, 0.25, 0.25, 0.25)) * 100
            elif references_llada:
                bleu2 = corpus_bleu(references_llada, hypotheses_llada, weights=(0.5, 0.5)) * 100
                bleu4 = corpus_bleu(references_llada, hypotheses_llada, weights=(0.25, 0.25, 0.25, 0.25)) * 100
            else:
                bleu2, bleu4 = 0.0, 0.0
        except ImportError:
            bleu2, bleu4 = 0.0, 0.0

        # METEOR - WordNet mode (word_tokenize, lowercase, ×100)
        meteor_score_wordnet = 0.0
        if use_wordnet and references_wordnet:
            try:
                from nltk.translate.meteor_score import meteor_score as nltk_meteor
                meteor_scores_wn = []
                for ref, hyp in zip(references_wordnet, hypotheses_wordnet):
                    meteor_scores_wn.append(nltk_meteor(ref, hyp))
                meteor_score_wordnet = float(np.mean(meteor_scores_wn)) * 100
            except ImportError:
                pass

        # METEOR - LLaDA mode (model tokenizer, ×100)
        meteor_score_llada = 0.0
        if use_llada and references_llada:
            try:
                from nltk.translate.meteor_score import meteor_score as nltk_meteor
                meteor_scores_ll = []
                for ref, hyp in zip(references_llada, hypotheses_llada):
                    meteor_scores_ll.append(nltk_meteor(ref, hyp))
                meteor_score_llada = float(np.mean(meteor_scores_ll)) * 100
            except ImportError:
                pass

        # ROUGE: no stemmer, score(hyp, ref) order, ×100 (matches Old_MolDA)
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
            rouge_scores = []
            for hyp_sen, ref_sen in zip(hyp_sentences, ref_sentences):
                lscore = scorer.score(hyp_sen, ref_sen)
                rouge_scores.append(lscore)
            rouge1 = float(np.mean([rs["rouge1"].fmeasure for rs in rouge_scores])) * 100
            rouge2 = float(np.mean([rs["rouge2"].fmeasure for rs in rouge_scores])) * 100
            rougeL = float(np.mean([rs["rougeL"].fmeasure for rs in rouge_scores])) * 100
        except ImportError:
            rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    else:
        bleu2, bleu4 = 0.0, 0.0
        meteor_score_wordnet = 0.0
        meteor_score_llada = 0.0
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0

    failure_rate = len(failure_idxs) / len(pred_texts) if len(pred_texts) > 0 else 0.0

    # Build results (matches Old_MolDA key structure)
    results = {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "failure_rate": failure_rate,
        "_failure_indices": failure_idxs,
    }

    # Add METEOR keys based on selected options
    if use_wordnet:
        results["meteor_wordnet"] = meteor_score_wordnet
    if use_llada:
        results["meteor_llada"] = meteor_score_llada

    # Backward-compatible "meteor" key (WordNet priority, then LLaDA)
    if use_wordnet:
        results["meteor"] = meteor_score_wordnet
    elif use_llada:
        results["meteor"] = meteor_score_llada
    else:
        results["meteor"] = 0.0

    return results


# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────

def evaluate_by_task(task: str, **kwargs) -> Dict[str, float]:
    """Dispatch to the appropriate evaluation function based on task type."""
    task_type = get_task_type(task)
    if task_type == "classification":
        return classification_evaluate(**kwargs, task=task)
    elif task_type == "regression":
        return regression_evaluate(**kwargs, task=task)
    elif task_type == "molecule":
        return molecule_evaluate(**kwargs, task=task)
    elif task_type == "caption":
        return caption_evaluate(**kwargs, task=task)
    elif task_type == "name_conversion":
        return {}  # MVP: skip
    else:
        return {}
