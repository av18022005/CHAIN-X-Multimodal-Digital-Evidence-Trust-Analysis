"""
src/fusion/nlp_crossmodal_check.py
Quick follow-up analysis: on the subset of images with REAL EXIF ground
truth (the 278 rows where Phase 4's consistency score is genuinely
meaningful, not just "unverifiable"), does the NLP report-consistency
score correlate with image classifier correctness?

Unlike the Phase 5 main test (metadata anomaly vs image score, tested on
all 3302 rows), this uses Phase 4's actual validated signal on its
genuinely-verifiable subset -- smaller sample, but zero synthetic-data
circularity concerns since consistency_score there was never used to
train anything.

Usage:
    python -m src.fusion.nlp_crossmodal_check \
        --image-scores features/image_scores.csv \
        --consistency-csv features/consistency_scores.csv \
        --out features/nlp_crossmodal_analysis.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-scores", type=Path, required=True)
    ap.add_argument("--consistency-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    img = pd.read_csv(args.image_scores)
    cons = pd.read_csv(args.consistency_csv)

    df = img.merge(
        cons[["case_id", "consistency_score", "flagged_contradiction", "ground_truth_available"]],
        on="case_id", how="inner",
    )

    # Only rows with REAL ground truth (not fabricated/unverifiable claims)
    df = df[df["ground_truth_available"] == True]  # noqa: E712
    test_df = df[df["split"].isin(["internal_test", "external_test"])].dropna(subset=["image_correct"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Rows with real ground truth: {len(df)} (all splits)")
    print(f"Rows with real ground truth in test splits: {len(test_df)}")

    if len(test_df) < 10:
        print("\nToo few rows in the test-split + real-ground-truth intersection "
              "for a meaningful statistical test. Report the sample size honestly "
              "and treat this as descriptive only, not a hypothesis test.")
        print(test_df[["case_id", "consistency_score", "image_correct"]])
        return

    correct_group = test_df.loc[test_df["image_correct"] == True, "consistency_score"]  # noqa: E712
    wrong_group = test_df.loc[test_df["image_correct"] == False, "consistency_score"]  # noqa: E712

    print(f"\nn(correct)={len(correct_group)}  median consistency={correct_group.median():.3f}")
    print(f"n(wrong)  ={len(wrong_group)}  median consistency={wrong_group.median():.3f}")

    if len(correct_group) > 0 and len(wrong_group) > 0:
        stat, p_value = mannwhitneyu(wrong_group, correct_group, alternative="two-sided")
        n1, n2 = len(wrong_group), len(correct_group)
        r = 1 - (2 * stat) / (n1 * n2)
        print(f"\nMann-Whitney U (two-sided): U={stat:.1f}, p={p_value:.4f}, rank-biserial r={r:.3f}")
        if p_value < 0.05:
            print("=> SIGNIFICANT association between report consistency and image classifier correctness.")
        else:
            print("=> NOT significant -- likely underpowered given the small real-ground-truth "
                  "sample size, or a genuine null result. Report the sample size alongside the "
                  "p-value so readers can judge for themselves.")
    else:
        print("One group is empty -- cannot run the test.")


if __name__ == "__main__":
    main()
