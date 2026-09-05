"""
src/fusion/extra_metrics.py
Adds dashboard-friendly derived metrics on top of the Evidence Health
Index output. These are presentation/triage metrics, not new statistical
findings -- they repackage existing validated + diagnostic signals into
forms more useful for a human reading a case report.

IMPORTANT: validated and diagnostic-only signals are kept strictly
separate -- no metric here blends a real (image/metadata) signal with
a synthetic-ground-truth (NLP/custody) signal. This mirrors the paper's
own split: only image + metadata feed the trained fusion model; NLP
consistency and custody integrity are demonstrated capabilities on
synthetic evaluation data, shown separately in the dashboard.

VALIDATED (real signals only -- safe to use in the paper too):
    - cross_modal_agreement_score: do image + metadata (the two REAL,
      validated signals) independently point the same direction? Built
      from the same z-score logic as Phase 5's disagreement_score, just
      inverted into a 0-1 "agreement" framing.
    - confidence_score: how far the fusion probability sits from 0.5 --
      a borderline case near 0.5 needs human review; one near 0 or 1
      doesn't.
    - case_priority_score: triage ranking derived ONLY from
      evidence_health_index (1 - health index, since higher health =
      lower priority to review). No diagnostic-only signal is mixed in.

DIAGNOSTIC / DEMO ONLY (synthetic ground truth -- label clearly in any
dashboard panel, never cite as a paper result):
    - documentation_reliability_score: dashboard-friendly rename of
      Phase 4's consistency_score.
    - custody_integrity_score: inverse of Phase 6's custody_risk_score.

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

    # ================= VALIDATED (real signals only) =================

    # --- Cross-modal agreement (image + metadata only) ---
    img_z = (df["image_tamper_prob"] - df["image_tamper_prob"].mean()) / df["image_tamper_prob"].std()
    meta_z = (df["metadata_anomaly_score"] - df["metadata_anomaly_score"].mean()) / df["metadata_anomaly_score"].std()
    disagreement = (img_z - meta_z).abs()
    # Squash to 0-1 with a smooth decay; higher disagreement -> lower agreement
    df["cross_modal_agreement_score"] = 1 / (1 + disagreement)

    # --- Confidence: distance of fusion probability from the maximally-uncertain midpoint ---
    df["confidence_score"] = (df["fusion_tamper_prob"] - 0.5).abs() * 2  # 0 = coin flip, 1 = fully confident

    # --- Triage priority: ONLY from the validated evidence health index ---
    # High health index = trustworthy/low-risk -> LOW priority to review.
    # Low health index = suspicious/high-risk -> HIGH priority to review.
    df["case_priority_score"] = (1 - df["evidence_health_index"]).clip(0, 1)

    # ============= DIAGNOSTIC / DEMO ONLY (synthetic GT) ==============

    df["documentation_reliability_score"] = df["consistency_score"]  # only meaningful where ground_truth_available
    df["custody_integrity_score"] = 1 - df["custody_risk_score"].clip(0, 1)

    out_cols = [
        "case_id", "split", "true_label",
        # validated
        "evidence_health_index", "cross_modal_agreement_score",
        "confidence_score", "case_priority_score",
        # diagnostic / demo only
        "documentation_reliability_score", "custody_integrity_score",
        "report_flag", "custody_flag",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.out, index=False)
    print(f"Wrote dashboard metrics for {len(df)} cases -> {args.out}")

    print("\n--- Validated metrics (real signals only) ---")
    for col in ["cross_modal_agreement_score", "confidence_score", "case_priority_score"]:
        print(f"  {col}: mean={df[col].mean():.3f}  min={df[col].min():.3f}  max={df[col].max():.3f}")

    print("\n--- Diagnostic / demo metrics (synthetic ground truth) ---")
    for col in ["custody_integrity_score"]:
        print(f"  {col}: mean={df[col].mean():.3f}  min={df[col].min():.3f}  max={df[col].max():.3f}")
    gt_rows = df[df["ground_truth_available"] == True] if "ground_truth_available" in df.columns else None  # noqa: E712
    if gt_rows is not None and len(gt_rows) > 0:
        col = "documentation_reliability_score"
        print(f"  {col} (n={len(gt_rows)}, GT-available rows only): "
              f"mean={gt_rows[col].mean():.3f}  min={gt_rows[col].min():.3f}  max={gt_rows[col].max():.3f}")


if __name__ == "__main__":
    main()
