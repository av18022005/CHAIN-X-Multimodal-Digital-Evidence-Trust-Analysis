"""
src/fusion/cross_modal_consistency.py
Phase 5, step 2: combine the image tamper score (Phase 2) with the metadata
anomaly score (Phase 3) into a single disagreement metric, then test the
actual research hypothesis:

    "Does cross-modal disagreement predict where the image classifier
     is more likely to be WRONG?"

This is the testable claim worth putting in a paper -- not just a fusion
score, but evidence for *why* combining modalities should help.

Method:
1. Z-score both signals using TRAIN-split statistics only (avoids leaking
   test-set distribution info into the "disagreement" computation).
2. disagreement_score = |z(image_tamper_prob) - z(metadata_anomaly_score)|
3. Split test-set rows into "image_correct" vs "image_wrong" (from Phase 5
   step 1's image_correct column).
4. Run a Mann-Whitney U test comparing disagreement_score between those two
   groups (non-parametric -- doesn't assume the scores are normally
   distributed, which anomaly/probability scores usually aren't).

A significant result (distribution of disagreement is higher when the image
classifier is wrong) is direct evidence that metadata provides an
independent, complementary signal -- exactly the justification a fusion
model (Phase 7) needs.

Usage:
    python -m src.fusion.cross_modal_consistency \
        --image-scores features/image_scores.csv \
        --metadata-csv features/metadata_features.csv \
        --out features/cross_modal_analysis.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def zscore_with_train_stats(series: pd.Series, train_mask: pd.Series) -> pd.Series:
    mu = series[train_mask].mean()
    sigma = series[train_mask].std()
    if sigma == 0 or pd.isna(sigma):
        sigma = 1.0  # avoid divide-by-zero on a degenerate split
    return (series - mu) / sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-scores", type=Path, required=True, help="Output of score_images.py")
    ap.add_argument("--metadata-csv", type=Path, required=True, help="Phase 3 metadata_features.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    img = pd.read_csv(args.image_scores)
    meta = pd.read_csv(args.metadata_csv)

    df = img.merge(meta[["case_id", "metadata_anomaly_score"]], on="case_id", how="inner")
    n_dropped = len(img) - len(df)
    if n_dropped:
        print(f"Warning: {n_dropped} rows had no matching metadata row and were dropped")

    train_mask = df["split"] == "train"
    if train_mask.sum() == 0:
        print("Warning: no 'train' split rows found -- falling back to whole-dataset stats "
              "for z-scoring (less rigorous, but still runs).")
        train_mask = pd.Series(True, index=df.index)

    df["image_z"] = zscore_with_train_stats(df["image_tamper_prob"], train_mask)
    df["metadata_z"] = zscore_with_train_stats(df["metadata_anomaly_score"], train_mask)
    df["disagreement_score"] = (df["image_z"] - df["metadata_z"]).abs()

    df["agreement_bucket"] = np.select(
        [
            (df["image_z"] > 0) & (df["metadata_z"] > 0),
            (df["image_z"] <= 0) & (df["metadata_z"] <= 0),
        ],
        ["both_high_suspicion", "both_low_suspicion"],
        default="disagreement",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")
    print("\nAgreement bucket counts:")
    print(df["agreement_bucket"].value_counts())

    # --- The actual hypothesis test ---
    if "image_correct" not in df.columns:
        print("\nNo 'image_correct' column found -- skipping hypothesis test "
              "(re-run score_images.py with a master CSV that has true labels).")
        return

    test_mask = df["split"].isin(["internal_test", "external_test"])
    test_df = df[test_mask].dropna(subset=["image_correct"])
    if test_df.empty:
        print("\nNo test-split rows with labels found -- skipping hypothesis test.")
        return

    correct_group = test_df.loc[test_df["image_correct"] == True, "disagreement_score"]  # noqa: E712
    wrong_group = test_df.loc[test_df["image_correct"] == False, "disagreement_score"]  # noqa: E712

    print(f"\n--- Hypothesis test: is disagreement higher when the image classifier is WRONG? ---")
    print(f"n(correct)={len(correct_group)}  mean disagreement={correct_group.mean():.3f}")
    print(f"n(wrong)  ={len(wrong_group)}  mean disagreement={wrong_group.mean():.3f}")

    if len(correct_group) > 0 and len(wrong_group) > 0:
        stat, p_value = mannwhitneyu(wrong_group, correct_group, alternative="greater")
        print(f"\nMann-Whitney U test (one-sided: wrong > correct): U={stat:.1f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("=> SIGNIFICANT: disagreement is significantly higher when the image "
                  "classifier is wrong. This supports using metadata as a complementary "
                  "signal in a fusion model.")
        else:
            print("=> NOT significant at p<0.05. Report this honestly -- a null result "
                  "here is still a valid, useful finding for the paper (it would mean "
                  "the two modalities' errors aren't well correlated in a way this "
                  "simple z-score disagreement metric captures).")
    else:
        print("One of the groups is empty -- cannot run the test.")


if __name__ == "__main__":
    main()
