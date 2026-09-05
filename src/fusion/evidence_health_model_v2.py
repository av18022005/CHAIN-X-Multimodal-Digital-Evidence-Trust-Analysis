"""
src/fusion/evidence_health_model_v2.py
Fusion attempt #2: instead of blending Phase 2's already-collapsed
image_tamper_prob with Phase 3's already-collapsed metadata_anomaly_score
(both pre-weighted by hand/by a separate model), feed the RAW underlying
features -- all 14 ELA/noise features + the raw metadata flags -- into
ONE RandomForest. A learned model may extract more signal than two
independently hand-tuned scores glued together afterward.

Compare this against:
    - the v1 fusion (image_tamper_prob + metadata_anomaly_score, logistic regression)
    - the image-only baseline (Phase 2)

Usage:
    python -m src.fusion.evidence_health_model_v2 \
        --master-csv /kaggle/working/data/master_index.csv \
        --ela-csv /kaggle/working/features/ela_features_final.csv \
        --metadata-csv /kaggle/working/features/metadata_features.csv \
        --out features/fusion_v2_scores.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve

SEED = 42

ELA_FEATURE_COLS = [
    "ela_mean", "ela_std", "ela_max", "ela_p95", "ela_high_energy_ratio",
    "laplacian_var", "local_noise_std_mean", "local_noise_std_std", "edge_density",
    "patch_ela_mean_of_means", "patch_ela_std_of_means", "patch_ela_max_zscore",
    "patch_ela_max_minus_median", "patch_edge_std_of_means",
]
METADATA_FLAG_COLS = [
    "flag_exif_missing", "flag_editing_software", "flag_implausible_timestamp",
    "flag_no_camera_info", "metadata_anomaly_score",
]


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
    ap.add_argument("--master-csv", type=Path, required=True)
    ap.add_argument("--ela-csv", type=Path, required=True)
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    master = pd.read_csv(args.master_csv)
    ela = pd.read_csv(args.ela_csv)
    meta = pd.read_csv(args.metadata_csv)[["case_id"] + METADATA_FLAG_COLS]

    df = master.merge(ela, on="case_id", how="inner").merge(meta, on="case_id", how="left")

    feature_cols = ELA_FEATURE_COLS + METADATA_FLAG_COLS
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    y = (df["label"] == "tampered").astype(int)
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    internal_mask = (df["split"] == "internal_test").values
    external_mask = (df["split"] == "external_test").values

    X = df[feature_cols].values
    clf = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X[train_mask], y[train_mask].values)
    probs = clf.predict_proba(X)[:, 1]
    threshold = find_best_threshold(y[val_mask].values, probs[val_mask])

    print(f"Fusion v2 decision threshold (val-tuned): {threshold:.3f}\n")
    print("=== FUSION v2 (raw ELA + raw metadata flags, RandomForest) ===")
    internal_metrics = evaluate_split(y[internal_mask], probs[internal_mask], threshold, "internal_test")
    external_metrics = evaluate_split(y[external_mask], probs[external_mask], threshold, "external_test")

    print("\n=== Feature importances (top 8) ===")
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(8).to_string())

    print("\nCompare these F1/AUC numbers against:")
    print("  v1 fusion (image_tamper_prob + metadata_anomaly_score, logistic regression)")
    print("  baseline (image-only, Phase 2)")
    print("...printed in your earlier evidence_health_model.py run.")

    out = df[["case_id", "split", "label"]].copy()
    out["fusion_v2_prob"] = probs
    out["fusion_v2_pred"] = (probs >= threshold).astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
