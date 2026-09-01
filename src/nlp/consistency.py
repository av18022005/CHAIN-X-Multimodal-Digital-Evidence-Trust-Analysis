"""
src/nlp/consistency.py
Compares claimed metadata (extracted from report text) against the
REAL metadata pulled from the image itself (Phase 3 output), and
computes a per-image consistency score.

Assumes:
- extracted claims CSV has: claimed_camera, claimed_date, + a join key
  (filename or image_id)
- Phase 3 metadata CSV has: camera_model, datetime_original (date part),
  + the same join key
"""

import re
import argparse
import difflib
import pandas as pd

CAMERA_MATCH_THRESHOLD = 0.6  # fuzzy string similarity cutoff


def normalize_camera(name) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def normalize_date(date_str) -> str:
    """Keep just the YYYY-MM-DD portion, tolerant of datetime strings."""
    if not isinstance(date_str, str):
        return ""
    match = re.search(r"\d{4}-\d{2}-\d{2}", date_str.replace(":", "-", 2))
    return match.group(0) if match else date_str.strip()


def camera_similarity(a: str, b: str) -> float:
    a, b = normalize_camera(a), normalize_camera(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_row(claimed_camera, claimed_date, real_camera, real_date) -> dict:
    cam_sim = camera_similarity(claimed_camera, real_camera)
    camera_match = cam_sim >= CAMERA_MATCH_THRESHOLD

    real_date_norm = normalize_date(real_date)
    claimed_date_norm = normalize_date(claimed_date)
    date_match = bool(claimed_date_norm) and (claimed_date_norm == real_date_norm)

    # Consistency score: 1.0 = fully consistent, 0.0 = fully contradicted.
    # Weighted average; missing real data treated as "cannot verify" (neutral 0.5).
    parts = []
    if real_camera:
        parts.append(1.0 if camera_match else 0.0)
    if real_date_norm:
        parts.append(1.0 if date_match else 0.0)
    consistency_score = sum(parts) / len(parts) if parts else 0.5

    return {
        "camera_match": camera_match,
        "camera_similarity": round(cam_sim, 3),
        "date_match": date_match,
        "consistency_score": round(consistency_score, 3),
        "flagged_contradiction": consistency_score < 0.5,
    }


def run_consistency_check(claims_csv: str, metadata_csv: str, join_key: str) -> pd.DataFrame:
    claims = pd.read_csv(claims_csv)
    meta = pd.read_csv(metadata_csv)

    for col in (join_key,):
        if col not in claims.columns or col not in meta.columns:
            raise ValueError(f"Join key '{col}' missing from one of the input CSVs. "
                              f"claims cols: {list(claims.columns)} | meta cols: {list(meta.columns)}")

    merged = claims.merge(meta, on=join_key, how="left", suffixes=("_claim", "_meta"))

    results = merged.apply(
        lambda r: score_row(
            r.get("claimed_camera"),
            r.get("claimed_date"),
            r.get("camera_model"),
            r.get("datetime_original"),
        ),
        axis=1,
    ).apply(pd.Series)

    return pd.concat([merged, results], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Check report claims against real metadata")
    parser.add_argument("--claims-csv", required=True, help="Output of extractor.py")
    parser.add_argument("--metadata-csv", required=True, help="Phase 3 metadata_features.csv")
    parser.add_argument("--join-key", default="case_id")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = run_consistency_check(args.claims_csv, args.metadata_csv, args.join_key)
    result.to_csv(args.out, index=False)

    n_flagged = result["flagged_contradiction"].sum()
    print(f"Scored {len(result)} rows -> {args.out}")
    print(f"  Flagged as contradictions: {n_flagged} ({n_flagged/len(result)*100:.1f}%)")


if __name__ == "__main__":
    main()
