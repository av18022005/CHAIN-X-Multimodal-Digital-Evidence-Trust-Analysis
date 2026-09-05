"""
src/fusion/extra_metrics.py
Adds dashboard-friendly derived metrics on top of the Evidence Health
Index output. These are presentation/triage metrics, not new statistical
findings -- they repackage existing validated + diagnostic signals into
forms more useful for a human reading a case report.

    - cross_modal_agreement_score: do image + metadata (the two REAL,
      validated signals) independently point the same direction? Built
      from the same z-score logic as Phase 5's disagreement_score, just
      inverted into a 0-1 "agreement" framing.
    - confidence_score: how far the fusion probability sits from 0.5 --
      a borderline case near 0.5 needs human review; one near 0 or 1
      doesn't.
    - documentation_reliability_score: dashboard-friendly rename of
      Phase 4's consistency_score (diagnostic/demo signal, synthetic GT).
    - custody_integrity_score: inverse of Phase 6's custody_risk_score
      (diagnostic/demo signal, synthetic GT).
    - overall_case_priority_score: a composite TRIAGE aid combining all
      of the above for dashboard display. Explicitly NOT a validated
      classifier output -- weights are illustrative, not learned.

Usage:
    python -m src.fusion.extra_metrics \
        --health-index-csv features/evidence_health_index.csv \
        --out features/dashboard_metrics.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-index-csv", type=Path, required=True,
                     help="Output of evidence_health_model.py (Phase 7)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.health_index_csv)

    # --- Cross-modal agreement (real signals only: image + metadata) ---
    img_z = (df["image_tamper_prob"] - df["image_tamper_prob"].mean()) / df["image_tamper_prob"].std()
    meta_z = (df["metadata_anomaly_score"] - df["metadata_anomaly_score"].mean()) / df["metadata_anomaly_score"].std()
    disagreement = (img_z - meta_z).abs()
    # Squash to 0-1 with a smooth decay; higher disagreement -> lower agreement
    df["cross_modal_agreement_score"] = 1 / (1 + disagreement)

    # --- Confidence: distance of fusion probability from the maximally-uncertain midpoint ---
    df["confidence_score"] = (df["fusion_tamper_prob"] - 0.5).abs() * 2  # 0 = coin flip, 1 = fully confident

    # --- Diagnostic signals, renamed for dashboard readability ---
    df["documentation_reliability_score"] = df["consistency_score"]  # only meaningful where ground_truth_available
    df["custody_integrity_score"] = 1 - df["custody_risk_score"].clip(0, 1)

    # --- Composite triage score (dashboard-only, NOT a validated classifier output) ---
    # Illustrative weights: validated evidence health index dominates; diagnostic
    # signals nudge the score only when they're actually informative.
    doc_component = df["documentation_reliability_score"].where(df["ground_truth_available"] == True, 0.5)  # noqa: E712
    df["overall_case_priority_score"] = (
        0.6 * df["evidence_health_index"]
        + 0.2 * df["custody_integrity_score"]
        + 0.2 * doc_component
    ).clip(0, 1)

    out_cols = [
        "case_id", "split", "true_label",
        "evidence_health_index", "cross_modal_agreement_score", "confidence_score",
        "documentation_reliability_score", "custody_integrity_score",
        "overall_case_priority_score", "report_flag", "custody_flag",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.out, index=False)

    print(f"Wrote dashboard metrics for {len(df)} cases -> {args.out}")
    print("\nSummary stats:")
    for col in ["cross_modal_agreement_score", "confidence_score",
                "custody_integrity_score", "overall_case_priority_score"]:
        print(f"  {col}: mean={df[col].mean():.3f}  min={df[col].min():.3f}  max={df[col].max():.3f}")


if __name__ == "__main__":
    main()
