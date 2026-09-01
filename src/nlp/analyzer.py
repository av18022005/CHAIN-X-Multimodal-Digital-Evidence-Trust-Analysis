"""
src/nlp/analyzer.py
Phase 4 CLI entry point. Ties together:
  1. extractor.py   -> pulls claimed camera/date out of report text
  2. consistency.py -> compares claims vs real Phase 3 metadata

Then evaluates: does flagged_contradiction actually catch the
synthetic contradictions injected by report_generator.py?

Expects report_generator.py's CSV to include a ground-truth column
(default name: 'is_contradiction_injected', bool). If that column is
missing, the eval section is skipped and only the consistency CSV is produced.
"""

import argparse
import pandas as pd

from src.nlp.extractor import extract_from_csv
from src.nlp.consistency import run_consistency_check, score_row


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
    }


def evaluate(result: pd.DataFrame, ground_truth_col: str) -> dict:
    if ground_truth_col not in result.columns:
        return {}

    # Only evaluate rows that actually had real metadata to compare against.
    # Rows with no real EXIF got a fabricated claim with no true right answer,
    # so including them would distort precision/recall.
    if "ground_truth_available" in result.columns:
        result = result[result["ground_truth_available"] == True]  # noqa: E712
        if result.empty:
            return {}

    metrics = {"overall": compute_binary_metrics(result[ground_truth_col], result["flagged_contradiction"])}

    # Per-field breakdown: does the system separately catch a wrong camera
    # vs. a wrong date? camera_match/date_match True = "system thinks this
    # field is consistent", so a caught contradiction is (not camera_match).
    if "camera_contradiction_injected" in result.columns and "camera_match" in result.columns:
        metrics["camera_field"] = compute_binary_metrics(
            result["camera_contradiction_injected"], ~result["camera_match"].astype(bool)
        )
    if "date_contradiction_injected" in result.columns and "date_match" in result.columns:
        metrics["date_field"] = compute_binary_metrics(
            result["date_contradiction_injected"], ~result["date_match"].astype(bool)
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 4: NLP/Report consistency analyzer")
    parser.add_argument("--reports-csv", required=True, help="Output of report_generator.py")
    parser.add_argument("--metadata-csv", required=True, help="Phase 3 metadata_features.csv")
    parser.add_argument("--text-col", default="report_text")
    parser.add_argument("--join-key", default="case_id")
    parser.add_argument("--ground-truth-col", default="is_contradiction_injected")
    parser.add_argument("--out", required=True, help="Final consistency-scored CSV path")
    args = parser.parse_args()

    # Step 1: extract claims from report text
    claims = extract_from_csv(args.reports_csv, args.text_col)
    claims_tmp_path = args.out.replace(".csv", "_claims_tmp.csv")
    claims.to_csv(claims_tmp_path, index=False)
    print(f"[1/3] Extracted claims for {len(claims)} reports")

    # Step 2: score consistency against real metadata
    result = run_consistency_check(claims_tmp_path, args.metadata_csv, args.join_key)
    result.to_csv(args.out, index=False)
    n_flagged = result["flagged_contradiction"].sum()
    print(f"[2/3] Consistency scored -> {args.out}")
    print(f"      Flagged as contradictions: {n_flagged} ({n_flagged/len(result)*100:.1f}%)")

    # Step 3: evaluate against injected ground truth, if available
    metrics = evaluate(result, args.ground_truth_col)
    if metrics:
        print(f"[3/3] Evaluation vs. injected ground truth ('{args.ground_truth_col}'):")
        for section, section_metrics in metrics.items():
            print(f"      -- {section} --")
            for k, v in section_metrics.items():
                print(f"         {k}: {v}")
    else:
        print(f"[3/3] No ground-truth column '{args.ground_truth_col}' found — skipping eval.")


if __name__ == "__main__":
    main()
