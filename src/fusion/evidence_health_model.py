"""
src/fusion/evidence_health_model.py
Phase 7: fusion model + Evidence Health Index.

IMPORTANT DESIGN NOTE (documented here so it isn't lost/misread later):
Only image_tamper_prob (Phase 2) and metadata_anomaly_score (Phase 3) are
computed from real CASIA data and correlate with the true tampering label.
The NLP consistency score (Phase 4) and custody risk score (Phase 6) are
EVALUATION-ONLY synthetic signals -- they were generated independently of
whether an image is actually tampered, purely to test whether those
checkers catch injected contradictions/anomalies. They have no real
relationship to `label` in this dataset by construction.

So this script does two separate, honest things:
    1. Trains a FUSION classifier on the two real, label-correlated
       signals (image + metadata) and compares it against an image-only
       baseline -- this is the actual "does fusion help" experiment.
    2. Builds an Evidence Health Index (EHI): a per-case report combining
       the TRAINED fusion probability (the real predictive signal) with
       the NLP and custody signals attached as separate DIAGNOSTIC FLAGS
       rather than blended into the trained probability. This mirrors how
       a real analyst would read a case report: "tampering probability is
       X, AND separately here are other red flags worth a human look."

Usage:
    python -m src.fusion.evidence_health_model \
        --image-scores features/image_scores.csv \
        --metadata-csv features/metadata_features.csv \
        --consistency-csv features/consistency_scores.csv \
        --custody-csv features/custody_risk_scores.csv \
        --out features/evidence_health_index.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve

SEED = 42


def find_best_threshold(y_true, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def evaluate_split(y_true, probs, threshold, name):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    print(f"  [{name}] n={len(y_true)}  Accuracy={acc:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-scores", type=Path, required=True)
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--consistency-csv", type=Path, required=True)
    ap.add_argument("--custody-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    img = pd.read_csv(args.image_scores)
    meta = pd.read_csv(args.metadata_csv)[["case_id", "metadata_anomaly_score"]]
    nlp = pd.read_csv(args.consistency_csv)[
        ["case_id", "consistency_score", "flagged_contradiction", "ground_truth_available"]
    ]
    custody = pd.read_csv(args.custody_csv)[
        ["case_id", "custody_risk_score", "flagged_custody_anomaly"]
    ]

    df = img.merge(meta, on="case_id", how="left") \
            .merge(nlp, on="case_id", how="left") \
            .merge(custody, on="case_id", how="left")

    y = (df["true_label"] == "tampered").astype(int)
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    internal_mask = (df["split"] == "internal_test").values
    external_mask = (df["split"] == "external_test").values

    # ---------- 1. Fusion classifier: image + metadata, vs. image-only baseline ----------
    fusion_features = ["image_tamper_prob", "metadata_anomaly_score"]
    X = df[fusion_features].fillna(df[fusion_features].mean()).values

    clf = LogisticRegression(random_state=SEED, class_weight="balanced")
    clf.fit(X[train_mask], y[train_mask].values)
    fusion_probs = clf.predict_proba(X)[:, 1]
    fusion_threshold = find_best_threshold(y[val_mask].values, fusion_probs[val_mask])

    print(f"Fusion model coefficients: image_tamper_prob={clf.coef_[0][0]:.3f}, "
          f"metadata_anomaly_score={clf.coef_[0][1]:.3f}")
    print(f"Fusion decision threshold (val-tuned): {fusion_threshold:.3f}\n")

    print("=== FUSION (image + metadata) ===")
    fusion_internal = evaluate_split(y[internal_mask], fusion_probs[internal_mask], fusion_threshold, "internal_test")
    fusion_external = evaluate_split(y[external_mask], fusion_probs[external_mask], fusion_threshold, "external_test")

    print("\n=== BASELINE (image only, from Phase 2/5) ===")
    image_probs = df["image_tamper_prob"].values
    image_threshold = find_best_threshold(y[val_mask].values, image_probs[val_mask])
    baseline_internal = evaluate_split(y[internal_mask], image_probs[internal_mask], image_threshold, "internal_test")
    baseline_external = evaluate_split(y[external_mask], image_probs[external_mask], image_threshold, "external_test")

    print("\n=== Fusion vs. baseline delta (F1) ===")
    print(f"  internal_test: {fusion_internal['f1'] - baseline_internal['f1']:+.3f}")
    print(f"  external_test: {fusion_external['f1'] - baseline_external['f1']:+.3f}")

    # ---------- 2. Evidence Health Index: trained probability + diagnostic flags ----------
    df["fusion_tamper_prob"] = fusion_probs
    df["fusion_pred"] = (fusion_probs >= fusion_threshold).astype(int)
    df["evidence_health_index"] = 1 - fusion_probs  # higher = more trustworthy

    # Diagnostic flags -- NOT blended into the trained probability (see module docstring).
    df["report_flag"] = df["flagged_contradiction"].fillna(False)
    df["custody_flag"] = df["flagged_custody_anomaly"].fillna(False)
    df["any_diagnostic_flag"] = df["report_flag"] | df["custody_flag"]

    out_cols = [
        "case_id", "split", "true_label",
        "image_tamper_prob", "metadata_anomaly_score", "fusion_tamper_prob", "fusion_pred",
        "evidence_health_index",
        "consistency_score", "ground_truth_available", "report_flag",
        "custody_risk_score", "custody_flag",
        "any_diagnostic_flag",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.out, index=False)
    print(f"\nWrote Evidence Health Index for {len(df)} cases -> {args.out}")
    print(f"Cases with a diagnostic flag (report or custody): {df['any_diagnostic_flag'].sum()} "
          f"({df['any_diagnostic_flag'].mean()*100:.1f}%) -- note: these flags are demonstrated "
          f"detection capability on synthetic ground truth, not evidence the flagged CASIA images "
          f"are actually tampered (see module docstring).")


if __name__ == "__main__":
    main()
