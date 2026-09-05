"""
src/fusion/evidence_health_model.py
Phase 7 (v1): trains a REAL fusion classifier on the two validated,
label-correlated signals only -- image_tamper_prob (Phase 2) and
metadata_anomaly_score (Phase 3) -- using logistic regression.

NLP consistency (Phase 4) and custody risk (Phase 6) are NOT used to
train anything: both were generated against synthetic, injected ground
truth that is independent of the real CASIA `label`, so training on
them would add noise (or manufacture a fake correlation). They are
attached to the output only as separate diagnostic flag columns, for
dashboard/report display -- never blended into the trained probability.

This is a reconstruction of the original v1 script (the file was lost
from the repo) -- rebuilt from the printed run output: coefficients
(image_tamper_prob ~11.5, metadata_anomaly_score ~1.9), val-tuned
threshold ~0.383, and the internal/external test F1/AUC numbers. Run
it and compare the printed numbers against those -- if they land close
to the original run, the reconstruction is faithful enough to treat as
the paper's official Phase 7 result again.

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-scores", type=Path, required=True,
                     help="Output of score_images.py (Phase 5) -- case_id, split, true_label, image_tamper_prob")
    ap.add_argument("--metadata-csv", type=Path, required=True,
                     help="Output of metadata analyzer.py (Phase 3)")
    ap.add_argument("--consistency-csv", type=Path, required=True,
                     help="Output of nlp analyzer.py (Phase 4) -- diagnostic only, not trained on")
    ap.add_argument("--custody-csv", type=Path, required=True,
                     help="Output of graph_analyzer.py (Phase 6) -- diagnostic only, not trained on")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    img = pd.read_csv(args.image_scores)
    meta = pd.read_csv(args.metadata_csv)
    cons = pd.read_csv(args.consistency_csv)
    custody = pd.read_csv(args.custody_csv)

    df = img.merge(meta[["case_id", "metadata_anomaly_score"]], on="case_id", how="left")

    # --- Diagnostic-only attachments (never fed into training) ---
    cons_cols = {"case_id": "case_id"}
    if "consistency_score" in cons.columns:
        cons_cols["consistency_score"] = "consistency_score"
    if "flagged_contradiction" in cons.columns:
        cons_cols["flagged_contradiction"] = "report_flag"
    if "ground_truth_available" in cons.columns:
        cons_cols["ground_truth_available"] = "ground_truth_available"
    cons_subset = cons[list(cons_cols.keys())].rename(columns=cons_cols)
    df = df.merge(cons_subset, on="case_id", how="left")

    custody_cols = {"case_id": "case_id"}
    if "custody_risk_score" in custody.columns:
        custody_cols["custody_risk_score"] = "custody_risk_score"
    flag_col = "flagged_anomaly" if "flagged_anomaly" in custody.columns else None
    if flag_col:
        custody_cols[flag_col] = "custody_flag"
    custody_subset = custody[list(custody_cols.keys())].rename(columns=custody_cols)
    df = df.merge(custody_subset, on="case_id", how="left")

    # true_label as 0/1 (assumes "authentic"/"tampered" strings; adjust if numeric already)
    if df["true_label"].dtype == object:
        y = (df["true_label"].str.lower() == "tampered").astype(int)
    else:
        y = df["true_label"].astype(int)
    df["_y"] = y

    feat_cols = ["image_tamper_prob", "metadata_anomaly_score"]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    internal_mask = df["split"] == "internal_test"
    external_mask = df["split"] == "external_test"

    clf = LogisticRegression()
    clf.fit(df.loc[train_mask, feat_cols], df.loc[train_mask, "_y"])

    coefs = dict(zip(feat_cols, clf.coef_[0]))
    print(f"Fusion model coefficients: image_tamper_prob={coefs['image_tamper_prob']:.3f}, "
          f"metadata_anomaly_score={coefs['metadata_anomaly_score']:.3f}")

    df["fusion_tamper_prob"] = clf.predict_proba(df[feat_cols])[:, 1]

    # Val-tune threshold: sweep and pick the one maximizing F1 on validation split
    val_probs = df.loc[val_mask, "fusion_tamper_prob"].values
    val_y = df.loc[val_mask, "_y"].values
    best_thr, best_f1 = 0.5, -1
    for thr in np.arange(0.05, 0.95, 0.001):
        f1 = f1_score(val_y, (val_probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"Fusion decision threshold (val-tuned): {best_thr:.3f}")

    print("\n=== FUSION (image + metadata) ===")
    for name, mask in [("internal_test", internal_mask), ("external_test", external_mask)]:
        m = evaluate(df.loc[mask, "_y"], df.loc[mask, "fusion_tamper_prob"], best_thr)
        print(f"  [{name}] n={m['n']}  Accuracy={m['accuracy']:.3f}  F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}")

    print("\n=== BASELINE (image only, from Phase 2/5) ===")
    # Reuse the same threshold-tuning approach on image_tamper_prob alone, on val split
    val_img = df.loc[val_mask, "image_tamper_prob"].values
    best_img_thr, best_img_f1 = 0.5, -1
    for thr in np.arange(0.05, 0.95, 0.001):
        f1 = f1_score(val_y, (val_img >= thr).astype(int), zero_division=0)
        if f1 > best_img_f1:
            best_img_f1, best_img_thr = f1, thr
    baseline_deltas = {}
    for name, mask in [("internal_test", internal_mask), ("external_test", external_mask)]:
        m = evaluate(df.loc[mask, "_y"], df.loc[mask, "image_tamper_prob"], best_img_thr)
        print(f"  [{name}] n={m['n']}  Accuracy={m['accuracy']:.3f}  F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}")
        baseline_deltas[name] = m["f1"]

    print("\n=== Fusion vs. baseline delta (F1) ===")
    for name, mask in [("internal_test", internal_mask), ("external_test", external_mask)]:
        fm = evaluate(df.loc[mask, "_y"], df.loc[mask, "fusion_tamper_prob"], best_thr)
        print(f"  {name}: {fm['f1'] - baseline_deltas[name]:+.3f}")

    # evidence_health_index: HIGH = trustworthy/low-risk (inverse of tamper probability)
    df["evidence_health_index"] = 1 - df["fusion_tamper_prob"]

    out_cols = [
        "case_id", "split", "true_label",
        "image_tamper_prob", "metadata_anomaly_score", "fusion_tamper_prob",
        "evidence_health_index",
        "consistency_score", "ground_truth_available", "report_flag",
        "custody_risk_score", "custody_flag",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.out, index=False)

    n_flagged = 0
    if "report_flag" in df.columns:
        n_flagged = ((df["report_flag"].fillna(False)) | (df.get("custody_flag", False).fillna(False) if "custody_flag" in df.columns else False)).sum()
    pct = 100 * n_flagged / len(df) if len(df) else 0
    print(f"\nWrote Evidence Health Index for {len(df)} cases -> {args.out}")
    print(f"Cases with a diagnostic flag (report or custody): {n_flagged} ({pct:.1f}%) "
          f"-- note: these flags are demonstrated detection capability on synthetic ground truth, "
          f"not evidence the flagged CASIA images are actually tampered (see module docstring).")


if __name__ == "__main__":
    main()
