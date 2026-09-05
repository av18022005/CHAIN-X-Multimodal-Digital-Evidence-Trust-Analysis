"""
src/fusion/score_images.py
Phase 5, step 1: run the locked-in Phase 2 image classifier over every row
and save the continuous tamper PROBABILITY (not just a hard label) per
case_id. Cross-modal fusion works better on continuous scores than on
already-thresholded predictions -- thresholding throws away information
this analysis needs.

Usage:
    python -m src.fusion.score_images \
        --master-csv /kaggle/working/data/master_index.csv \
        --ela-csv /kaggle/working/features/ela_features_final.csv \
        --model-path models/rf_final_locked_in.pkl \
        --threshold 0.447 \
        --out features/image_scores.csv
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, required=True)
    ap.add_argument("--ela-csv", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.447,
                     help="Validation-tuned decision threshold from the classifier training run "
                          "(printed as '[threshold] Using validation-tuned decision threshold = ...' "
                          "when you trained rf_final_locked_in.pkl). NOT saved inside the pickle, "
                          "so it must be passed in manually -- check your training run's output.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.model_path, "rb") as f:
        saved = pickle.load(f)
    clf = saved["model"]
    feature_cols = saved["feature_cols"]

    master = pd.read_csv(args.master_csv)
    ela = pd.read_csv(args.ela_csv)
    df = master.merge(ela, on="case_id", how="inner")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ELA CSV is missing expected feature columns: {missing}")

    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    probs = clf.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "case_id": df["case_id"],
        "split": df.get("split"),
        "true_label": df.get("label"),
        "image_tamper_prob": probs,
        "image_pred": (probs >= args.threshold).astype(int),
    })
    if "label" in df.columns:
        out["image_correct"] = (
            (out["image_pred"] == 1) == (out["true_label"] == "tampered")
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} images -> {args.out}")
    if "image_correct" in out.columns:
        print(f"Overall accuracy at threshold {args.threshold}: {out['image_correct'].mean():.3f}")


if __name__ == "__main__":
    main()
