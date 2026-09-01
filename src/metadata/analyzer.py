"""
Phase 3 entry point: extract EXIF/file metadata and compute anomaly scores
for every image in the master CSV. Runs entirely on CPU.

Usage:
    python -m src.metadata.analyzer --master-csv data/master_index.csv \
        --out features/metadata_features.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.metadata.extractor import extract_all_metadata
from src.metadata.anomaly import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, default=Path("data/master_index.csv"))
    ap.add_argument("--out", type=Path, default=Path("features/metadata_features.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.master_csv)
    rows = []
    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Metadata extraction"):
        try:
            meta = extract_all_metadata(row["image_path"])
            result = analyze(meta)
            result["case_id"] = row["case_id"]
            # Keep a few raw fields alongside the flags/score for inspection
            result["exif_present"] = meta["exif_present"]
            result["camera_make"] = meta["camera_make"]
            result["camera_model"] = meta["camera_model"]
            result["software"] = meta["software"]
            result["datetime_original"] = meta["datetime_original"]
            rows.append(result)
        except Exception as e:
            failed.append((row["case_id"], str(e)))

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    if failed:
        print(f"{len(failed)} images failed, e.g.: {failed[:5]}")

    # Quick summary so you can sanity-check the result immediately
    print("\n--- Summary ---")
    print(f"EXIF present: {out_df['exif_present'].sum()} / {len(out_df)}")
    print(f"Editing software detected: {out_df['flag_editing_software'].sum()}")
    print(f"Implausible timestamps: {out_df['flag_implausible_timestamp'].sum()}")
    print(f"Mean anomaly score: {out_df['metadata_anomaly_score'].mean():.3f}")

    # Compare anomaly score by true label, if the master CSV has labels
    # (useful sanity check: do tampered images actually score higher?)
    merged = df.merge(out_df[["case_id", "metadata_anomaly_score"]], on="case_id", how="inner")
    if "label" in merged.columns:
        print("\nMean anomaly score by true label:")
        print(merged.groupby("label")["metadata_anomaly_score"].mean())


if __name__ == "__main__":
    main()
