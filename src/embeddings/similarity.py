"""
similarity.py — Cosine-similarity scoring for embedding-based matching features.

Builds on embed.py's Embedder (which returns L2-normalized vectors, so cosine
similarity == dot product). Adds:

  1. Pairwise + batch cosine similarity helpers.
  2. A ScriptAwareThreshold: because our own smoke testing showed cosine sim
     is NOT uniformly calibrated across scripts (e.g. Bengali true matches
     landed at ~0.38, barely above negatives at ~0.27-0.34, while
     Devanagari/Urdu/Tamil true matches cleanly separated at 0.60-0.73),
     a single global threshold is unsafe. This lets you set per-script
     thresholds, or gate (disable) the embedding feature entirely for
     scripts that haven't been validated on real data.
  3. A feature-add function matching the ML-2 job spec: takes a candidate
     pairs table + text columns, returns cosine sim columns appended.

This module does not decide the final threshold for you — that must be
grid-searched against the real validation set per the job spec ("grid-search
the decision threshold against the official macro F0.5 implementation").
The defaults here are provisional, based on the anecdotal smoke test, and
MUST be re-validated once real labeled data exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # feature-add helpers require pandas; core sim functions don't


# --------------------------------------------------------------------------- #
# Core similarity functions
# --------------------------------------------------------------------------- #

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors. Works even if not pre-normalized."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of a (N,D) and rows of b (M,D).

    Returns an (N, M) matrix. Safe for non-normalized input.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return a_norm @ b_norm.T


def cosine_sim_pairs(a_vecs: np.ndarray, b_vecs: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for two (N, D) arrays -> (N,) scores.

    Use this for candidate-pair scoring: row i of a_vecs vs row i of b_vecs,
    NOT the full cross product (use cosine_sim_matrix for that).
    """
    a_vecs = np.asarray(a_vecs, dtype=np.float32)
    b_vecs = np.asarray(b_vecs, dtype=np.float32)
    if a_vecs.shape != b_vecs.shape:
        raise ValueError(f"Shape mismatch: {a_vecs.shape} vs {b_vecs.shape}")
    a_norm = a_vecs / np.clip(np.linalg.norm(a_vecs, axis=1, keepdims=True), 1e-8, None)
    b_norm = b_vecs / np.clip(np.linalg.norm(b_vecs, axis=1, keepdims=True), 1e-8, None)
    return np.sum(a_norm * b_norm, axis=1)


# --------------------------------------------------------------------------- #
# Script detection (lightweight — Unicode block based, no external deps)
# --------------------------------------------------------------------------- #

# Rough Unicode ranges for scripts relevant to Indian business names.
# Not exhaustive/linguistically rigorous — good enough to route a threshold.
_SCRIPT_RANGES = {
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A)],
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "urdu_arabic": [(0x0600, 0x06FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
}


def detect_script(text: str) -> str:
    """Best-guess dominant script of a string, by counting characters per Unicode block.

    Returns one of _SCRIPT_RANGES's keys, or "unknown" if no letters match / text is empty.
    Mixed-script strings (e.g. "ABC ट्रेडर्स") return whichever script has the most chars —
    reasonable for business names, which are rarely evenly mixed.
    """
    counts: Dict[str, int] = {k: 0 for k in _SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for script, ranges in _SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] += 1
                break
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "unknown"


def script_pair(text_a: str, text_b: str) -> str:
    """Label for a pair, e.g. 'latin-bengali'. Order-independent (sorted)."""
    s1, s2 = sorted([detect_script(text_a), detect_script(text_b)])
    return f"{s1}-{s2}"


# --------------------------------------------------------------------------- #
# Script-aware thresholding
# --------------------------------------------------------------------------- #

@dataclass
class ScriptAwareThreshold:
    """Per-script-pair similarity thresholds, with a gating mechanism.

    default_threshold: used for any script pair not explicitly listed.
    thresholds: explicit per-script-pair overrides, e.g. {"latin-tamil": 0.5}.
    gated_pairs: script pairs where the embedding feature should be IGNORED
        entirely (e.g. not passed to the classifier, or zeroed out) because
        it hasn't been shown to separate matches from non-matches reliably.
        Per our smoke test, "bengali-latin" is a candidate for gating until
        validated on real labeled data.

    IMPORTANT: the numbers below are PROVISIONAL, derived from a 3-negative,
    5-positive anecdotal smoke test — not a real validation set. Re-run
    calibrate() against real held-out data before trusting these in the
    final model, per the job spec's "never tune on the hidden leaderboard /
    use the held-out validation split" rule.
    """

    default_threshold: float = 0.45
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "latin-tamil": 0.50,
        "latin-urdu_arabic": 0.50,
        "latin-devanagari": 0.45,
    })
    gated_pairs: set = field(default_factory=lambda: {"bengali-latin"})

    def threshold_for(self, script_pair_label: str) -> Optional[float]:
        """Returns the threshold to use, or None if this pair is gated (feature disabled)."""
        if script_pair_label in self.gated_pairs:
            return None
        return self.thresholds.get(script_pair_label, self.default_threshold)

    def is_match(self, sim: float, text_a: str, text_b: str) -> Optional[bool]:
        """Returns True/False, or None if the embedding feature is gated for this script pair
        (caller should fall back to lexical/address features only in that case)."""
        pair = script_pair(text_a, text_b)
        thresh = self.threshold_for(pair)
        if thresh is None:
            return None
        return sim >= thresh

    @staticmethod
    def calibrate(
        sims: Sequence[float],
        labels: Sequence[int],
        script_pairs: Sequence[str],
    ) -> "ScriptAwareThreshold":
        """Grid-search a per-script-pair threshold that maximizes F0.5 on real
        labeled validation data. Call this once real data + labels exist —
        replaces the provisional defaults above.

        sims, labels, script_pairs must be equal-length, aligned by row.
        labels: 1 = true match, 0 = non-match.
        """
        if pd is None:
            raise ImportError("pandas required for calibrate()")

        df = pd.DataFrame({"sim": sims, "label": labels, "pair": script_pairs})
        thresholds: Dict[str, float] = {}
        gated: set = set()

        for pair, group in df.groupby("pair"):
            best_f05, best_t = -1.0, 0.5
            for t in np.arange(0.05, 0.96, 0.01):
                pred = (group["sim"] >= t).astype(int)
                tp = ((pred == 1) & (group["label"] == 1)).sum()
                fp = ((pred == 1) & (group["label"] == 0)).sum()
                fn = ((pred == 0) & (group["label"] == 1)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                beta2 = 0.5 ** 2
                f05 = (
                    (1 + beta2) * precision * recall / (beta2 * precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                if f05 > best_f05:
                    best_f05, best_t = f05, t

            # If even the best threshold for this pair can't beat a trivial
            # "always reject" baseline by a meaningful margin, gate it.
            if best_f05 < 0.05:
                gated.add(pair)
            else:
                thresholds[pair] = float(best_t)

        return ScriptAwareThreshold(thresholds=thresholds, gated_pairs=gated)


# --------------------------------------------------------------------------- #
# Feature-table integration (per ML-2 job spec: "add cosine similarities as
# features to the same classifier framework")
# --------------------------------------------------------------------------- #

def add_embedding_features(
    df,  # pandas DataFrame
    embedder,  # embed.Embedder instance
    name_col_a: str,
    name_col_b: str,
    address_col_a: Optional[str] = None,
    address_col_b: Optional[str] = None,
    prefix: str = "emb_",
):
    """Appends embedding-based cosine similarity columns to a candidate-pairs table.

    Adds:
      f"{prefix}name_cosine_sim"
      f"{prefix}address_cosine_sim"   (only if address columns given)
      f"{prefix}script_pair"          (for downstream gating / analysis)

    Does NOT drop or filter rows — gating decisions belong to the classifier
    stage (e.g. pass script_pair as a categorical feature, or null out the
    cosine sim column for gated pairs before training).
    """
    if pd is None:
        raise ImportError("pandas required for add_embedding_features()")

    name_a_vecs = embedder.encode(df[name_col_a].fillna("").tolist())
    name_b_vecs = embedder.encode(df[name_col_b].fillna("").tolist())
    df = df.copy()
    df[f"{prefix}name_cosine_sim"] = cosine_sim_pairs(name_a_vecs, name_b_vecs)
    df[f"{prefix}script_pair"] = [
        script_pair(a, b) for a, b in zip(df[name_col_a].fillna(""), df[name_col_b].fillna(""))
    ]

    if address_col_a and address_col_b:
        addr_a_vecs = embedder.encode(df[address_col_a].fillna("").tolist())
        addr_b_vecs = embedder.encode(df[address_col_b].fillna("").tolist())
        df[f"{prefix}address_cosine_sim"] = cosine_sim_pairs(addr_a_vecs, addr_b_vecs)

    return df


if __name__ == "__main__":
    # Smoke test using the exact numbers you already validated on Colab —
    # confirms detect_script / script_pair / gating logic behaves sensibly
    # without needing the model loaded again.
    examples = [
        ("ABC Traders", "एबीसी ट्रेडर्स", 0.5962),
        ("ABC Traders", "এবিসি ট্রেডার্স", 0.3798),
        ("ABC Traders", "ABC டிரேடர்ஸ்", 0.7311),
        ("ABC Traders", "اے بی سی ٹریڈرز", 0.6882),
        ("ABC Traders", "XYZ Pharmaceuticals Pvt Ltd", 0.2682),
        ("ABC Traders", "Sharma Electronics", 0.3387),
        ("ABC Traders", "Global Logistics Solutions", 0.3212),
    ]

    threshold_model = ScriptAwareThreshold()

    print(f"{'sim':>7}  {'script_pair':<20}  {'verdict':<10}  text_b")
    for text_a, text_b, sim in examples:
        pair = script_pair(text_a, text_b)
        verdict = threshold_model.is_match(sim, text_a, text_b)
        verdict_str = "GATED" if verdict is None else ("MATCH" if verdict else "no match")
        print(f"{sim:7.4f}  {pair:<20}  {verdict_str:<10}  {text_b}")