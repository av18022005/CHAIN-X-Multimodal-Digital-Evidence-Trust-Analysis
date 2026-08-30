"""
Combine ELA/noise features + EfficientNet embeddings -> RandomForest/XGBoost
tampering classifier. Reports accuracy/F1/ROC-AUC on CASIA internal test AND
on COVERAGE external test (never touched during training).

Usage:
    python -m src.image.classifier \
        --master-csv data/master_index.csv \
        --ela-csv features/ela_features.csv \
        --embeddings-dir features/embeddings \
        --model rf
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_recall_curve

SEED = 42


def find_best_threshold(y_true, probs):
    """Pick the probability threshold that maximizes F1 on the given
    (validation) set. Avoids blindly using 0.5, which can be a poor cutoff
    when classes are imbalanced or the model's calibration is off."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def assert_no_leakage(df: pd.DataFrame):
    """Refuse to proceed if any file hash appears in both an internal split
    (train/val/internal_test) and external_test — this is the CASIA vs
    COVERAGE leakage guard called out in the plan."""
    internal_hashes = set(df[df["split"] != "external_test"]["sha256"])
    external_hashes = set(df[df["split"] == "external_test"]["sha256"])
    overlap = internal_hashes & external_hashes
    overlap.discard("")  # in case hashing was skipped
    if overlap:
        raise RuntimeError(
            f"Data leakage detected: {len(overlap)} file(s) appear in both "
            f"internal and external splits. Aborting."
        )


def load_embedding(embeddings_dir: Path, case_id: str, dim: int = 1280) -> np.ndarray:
    path = embeddings_dir / f"{case_id}.npy"
    if path.exists():
        return np.load(path)
    return np.full(dim, np.nan)  # missing embedding -> handled by caller


def build_feature_table(master_csv, ela_csv, embeddings_dir, feature_set="all"):
    df = pd.read_csv(master_csv)
    if "sha256" in df.columns:
        assert_no_leakage(df)

    ela_df = pd.read_csv(ela_csv)
    df = df.merge(ela_df, on="case_id", how="left")

    emb_cols = [f"emb_{i}" for i in range(1280)]
    embeddings = np.stack([load_embedding(embeddings_dir, cid) for cid in df["case_id"]])
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    ela_feature_cols = [c for c in ela_df.columns if c != "case_id"]

    # Drop rows with no embedding AND no ELA features at all (nothing to learn from)
    before = len(df)
    df = df.dropna(subset=emb_cols + ela_feature_cols, how="all")
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with no usable image features")

    if feature_set == "ela_only":
        feature_cols = ela_feature_cols
    elif feature_set == "embeddings_only":
        feature_cols = emb_cols
    else:
        feature_cols = ela_feature_cols + emb_cols

    # Simple imputation: fill remaining NaNs (partial modality availability) with column mean
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    return df, feature_cols


def train_and_eval(df, feature_cols, model_type="rf"):
    y = (df["label"] == "tampered").astype(int)
    X = df[feature_cols].values

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    internal_test_mask = df["split"] == "internal_test"
    external_test_mask = df["split"] == "external_test"

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1,
            class_weight="balanced",  # counter the 500 authentic / 800 tampered imbalance
        )
    elif model_type == "xgb":
        from xgboost import XGBClassifier
        y_train = y[train_mask.values]
        # scale_pos_weight = (negative count) / (positive count), computed on TRAIN only
        n_pos = int(y_train.sum())
        n_neg = int((1 - y_train).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=SEED, eval_metric="logloss", n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X[train_mask.values], y[train_mask.values])

    # Tune the decision threshold on VALIDATION only, then apply it everywhere else
    val_probs = clf.predict_proba(X[val_mask.values])[:, 1] if val_mask.sum() > 0 else None
    threshold = find_best_threshold(y[val_mask.values], val_probs) if val_probs is not None else 0.5
    print(f"\n[threshold] Using validation-tuned decision threshold = {threshold:.3f} (default would be 0.5)")

    def report(mask, name):
        if mask.sum() == 0:
            print(f"[{name}] no rows, skipping")
            return
        probs = clf.predict_proba(X[mask.values])[:, 1]
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y[mask.values], preds)
        f1 = f1_score(y[mask.values], preds)
        try:
            auc = roc_auc_score(y[mask.values], probs)
        except ValueError:
            auc = float("nan")  # e.g. only one class present
        print(f"\n=== {name} (n={mask.sum()}) ===")
        print(f"Accuracy={acc:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
        print(classification_report(y[mask.values], preds, target_names=["authentic", "tampered"]))
        return {"accuracy": acc, "f1": f1, "roc_auc": auc}

    results = {
        "val": report(val_mask, "VALIDATION (CASIA)"),
        "internal_test": report(internal_test_mask, "INTERNAL TEST (CASIA)"),
        "external_test": report(external_test_mask, "EXTERNAL TEST (COVERAGE)"),
    }
    return clf, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, default=Path("data/master_index.csv"))
    ap.add_argument("--ela-csv", type=Path, default=Path("features/ela_features.csv"))
    ap.add_argument("--embeddings-dir", type=Path, default=Path("features/embeddings"))
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    ap.add_argument("--features", choices=["all", "ela_only", "embeddings_only"], default="all",
                     help="Ablation: use only ELA features, only CNN embeddings, or both combined")
    ap.add_argument("--out-model", type=Path, default=None,
                     help="Defaults to models/image_classifier_<model>_<features>.pkl")
    args = ap.parse_args()

    if args.out_model is None:
        args.out_model = Path(f"models/image_classifier_{args.model}_{args.features}.pkl")

    df, feature_cols = build_feature_table(
        args.master_csv, args.ela_csv, args.embeddings_dir, feature_set=args.features
    )
    print(f"Feature table: {len(df)} rows, {len(feature_cols)} features ({args.features})")

    clf, results = train_and_eval(df, feature_cols, model_type=args.model)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "wb") as f:
        pickle.dump({"model": clf, "feature_cols": feature_cols}, f)
    print(f"\nSaved model to {args.out_model}")

    # Append to experiment log (Part of "track every experiment")
    log_path = Path("experiments/experiment_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment": args.out_model.stem,
        "dataset": "CASIA+COVERAGE",
        "model": args.model,
        "features": args.features,
        "val_f1": results["val"]["f1"] if results["val"] else None,
        "internal_test_f1": results["internal_test"]["f1"] if results["internal_test"] else None,
        "external_test_f1": results["external_test"]["f1"] if results["external_test"] else None,
        "internal_test_auc": results["internal_test"]["roc_auc"] if results["internal_test"] else None,
        "external_test_auc": results["external_test"]["roc_auc"] if results["external_test"] else None,
    }
    log_df = pd.DataFrame([row])
    if log_path.exists():
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
    print(f"Logged experiment to {log_path}")


if __name__ == "__main__":
    main()
