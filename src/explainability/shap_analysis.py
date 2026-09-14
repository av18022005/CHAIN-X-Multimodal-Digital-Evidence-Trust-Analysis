"""
src/explainability/shap_analysis.py
Phase 8: SHAP explainability for the locked-in Phase 2 image classifier
(rf_final_locked_in.pkl). Turns the RandomForest's raw predictions into
"why was THIS case flagged" explanations, using the same 14 ELA/noise/
patch features used to train it.

Two outputs:
1. Global feature importance (mean |SHAP value| across all cases) -- which
   forensic features matter most OVERALL for this model's decisions.
2. Per-case explanations for a handful of representative examples: a
   confidently-correct tampered case, a confidently-correct authentic
   case, and (if any exist) a misclassified case -- showing exactly which
   features pushed the decision which way.

Also saves the full per-case SHAP value table to CSV so Phase 9's
dashboard can show a "why was this flagged" breakdown for ANY case, not
just the handful printed here.

Usage:
    python -m src.explainability.shap_analysis \
        --master-csv /kaggle/working/data/master_index.csv \
        --ela-csv /kaggle/working/features/ela_features_final.csv \
        --model-path models/rf_final_locked_in.pkl \
        --out-importance features/shap_global_importance.csv \
        --out-values features/shap_case_values.csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap


def load_data(master_csv: Path, ela_csv: Path, feature_cols: list) -> pd.DataFrame:
    master = pd.read_csv(master_csv)
    ela = pd.read_csv(ela_csv)
    df = master.merge(ela, on="case_id", how="inner")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ELA CSV is missing expected feature columns: {missing}")

    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    return df


def get_positive_class_shap(shap_values, expected_value):
    """Different SHAP/sklearn version combinations return the binary-class
    result in different shapes. Normalize to a single (n_samples, n_features)
    array for the "tampered" (positive) class, plus its scalar base value."""
    if isinstance(shap_values, list):
        # Older SHAP API: list of two (n_samples, n_features) arrays, [class0, class1]
        values = shap_values[1]
        base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # Newer SHAP API: single (n_samples, n_features, n_classes) array
        values = shap_values[:, :, 1]
        base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        # Already a single (n_samples, n_features) array
        values = shap_values
        base = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    return values, float(base)


def print_case_explanation(case_row, shap_row, feature_cols, base_value, top_n=5):
    contributions = sorted(
        zip(feature_cols, shap_row, case_row[feature_cols].values),
        key=lambda x: abs(x[1]), reverse=True
    )[:top_n]
    pred_prob = case_row.get("_pred_prob", case_row.get("image_tamper_prob", "n/a"))
    print(f"  case_id={case_row['case_id']}  true_label={case_row.get('label', 'n/a')}  "
          f"predicted_prob={pred_prob:.3f}" if isinstance(pred_prob, float) else
          f"  case_id={case_row['case_id']}  true_label={case_row.get('label', 'n/a')}  "
          f"predicted_prob={pred_prob}")
    print(f"  base rate (average tamper log-odds contribution): {base_value:.3f}")
    for feat, shap_val, feat_val in contributions:
        direction = "-> pushes toward TAMPERED" if shap_val > 0 else "-> pushes toward AUTHENTIC"
        print(f"    {feat:30s} value={feat_val:10.3f}  SHAP={shap_val:+.3f}  {direction}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, default=Path("data/master_index.csv"))
    ap.add_argument("--ela-csv", type=Path, default=Path("features/ela_features_final.csv"))
    ap.add_argument("--model-path", type=Path, default=Path("models/rf_final_locked_in.pkl"))
    ap.add_argument("--out-importance", type=Path, default=Path("features/shap_global_importance.csv"))
    ap.add_argument("--out-values", type=Path, default=Path("features/shap_case_values.csv"))
    ap.add_argument("--n-examples", type=int, default=3,
                     help="How many example cases per category to print in detail")
    args = ap.parse_args()

    with open(args.model_path, "rb") as f:
        saved = pickle.load(f)
    clf = saved["model"]
    feature_cols = saved["feature_cols"]

    df = load_data(args.master_csv, args.ela_csv, feature_cols)
    X = df[feature_cols].values

    print(f"Computing SHAP values for {len(df)} cases, {len(feature_cols)} features...")
    explainer = shap.TreeExplainer(clf)
    raw_shap_values = explainer.shap_values(X)
    shap_values, base_value = get_positive_class_shap(raw_shap_values, explainer.expected_value)

    # --- 1. Global feature importance ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    args.out_importance.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(args.out_importance, index=False)
    print(f"\n=== Global feature importance (mean |SHAP|) ===")
    print(importance_df.to_string(index=False))
    print(f"\nSaved -> {args.out_importance}")

    # --- 2. Save full per-case SHAP table (for Phase 9 dashboard) ---
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    shap_df["case_id"] = df["case_id"].values
    shap_df["base_value"] = base_value
    args.out_values.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(args.out_values, index=False)
    print(f"Saved per-case SHAP values -> {args.out_values}")

    # --- 3. Example case explanations ---
    if "label" not in df.columns:
        print("\nNo 'label' column found -- skipping example case explanations.")
        return

    predicted_prob = clf.predict_proba(X)[:, 1]
    df = df.assign(_shap_idx=np.arange(len(df)), _pred_prob=predicted_prob)

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Confidently-correct TAMPERED case(s)")
    print("=" * 70)
    correct_tampered = df[(df["label"] == "tampered") & (df["_pred_prob"] > 0.8)] \
        .sort_values("_pred_prob", ascending=False).head(args.n_examples)
    for _, row in correct_tampered.iterrows():
        print_case_explanation(row, shap_values[row["_shap_idx"]], feature_cols, base_value)

    print("=" * 70)
    print("EXAMPLE 2: Confidently-correct AUTHENTIC case(s)")
    print("=" * 70)
    correct_authentic = df[(df["label"] == "authentic") & (df["_pred_prob"] < 0.2)] \
        .sort_values("_pred_prob", ascending=True).head(args.n_examples)
    for _, row in correct_authentic.iterrows():
        print_case_explanation(row, shap_values[row["_shap_idx"]], feature_cols, base_value)

    print("=" * 70)
    print("EXAMPLE 3: Misclassified case(s) (if any in this sample)")
    print("=" * 70)
    misclassified = df[
        ((df["label"] == "tampered") & (df["_pred_prob"] < 0.3)) |
        ((df["label"] == "authentic") & (df["_pred_prob"] > 0.7))
    ].head(args.n_examples)
    if misclassified.empty:
        print("  None found in this sample at the 0.3/0.7 confidence bar.")
    for _, row in misclassified.iterrows():
        print_case_explanation(row, shap_values[row["_shap_idx"]], feature_cols, base_value)


if __name__ == "__main__":
    main()
