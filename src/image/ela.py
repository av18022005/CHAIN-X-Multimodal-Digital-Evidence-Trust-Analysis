"""
Traditional forensic features: Error Level Analysis (ELA), noise statistics,
and simple texture/edge inconsistency signals.

Runs entirely on CPU. Computed on demand — NOT saved as extra image files
(per the resource-saving plan), only as numeric feature vectors appended to
the master CSV / a features parquet.

Usage:
    python -m src.image.ela --master-csv data/master_index.csv \
        --out features/ela_features.csv
"""
import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def compute_ela(path: str, quality: int = 90) -> np.ndarray:
    """Standard ELA: re-save the image at a known JPEG quality, take the
    absolute difference from the original. Returns a single-channel
    difference image (uint8)."""
    original = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    orig_arr = np.array(original, dtype=np.int16)
    resaved_arr = np.array(resaved, dtype=np.int16)
    diff = np.abs(orig_arr - resaved_arr).astype(np.uint8)
    return cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)


def ela_summary_features(ela_map: np.ndarray, prefix: str = "ela") -> dict:
    return {
        f"{prefix}_mean": float(np.mean(ela_map)),
        f"{prefix}_std": float(np.std(ela_map)),
        f"{prefix}_max": float(np.max(ela_map)),
        f"{prefix}_p95": float(np.percentile(ela_map, 95)),
        f"{prefix}_high_energy_ratio": float(np.mean(ela_map > 30)),  # fraction of "hot" pixels
    }


def noise_features(gray: np.ndarray) -> dict:
    """Simple high-frequency noise proxy via Laplacian variance, plus local
    noise-level std computed on a coarse grid (heuristic, not a full PRNU
    pipeline — good enough as a weak forensic signal for the fusion model)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    grid = 16
    h, w = gray.shape
    local_stds = []
    for y in range(0, h - grid, grid):
        for x in range(0, w - grid, grid):
            local_stds.append(gray[y:y + grid, x:x + grid].std())
    local_stds = np.array(local_stds) if local_stds else np.array([0.0])
    return {
        "laplacian_var": float(lap.var()),
        "local_noise_std_mean": float(local_stds.mean()),
        "local_noise_std_std": float(local_stds.std()),
    }


def edge_features(gray: np.ndarray) -> dict:
    edges = cv2.Canny(gray, 100, 200)
    return {
        "edge_density": float(np.mean(edges > 0)),
    }


def extract_all_features(path: str, qualities=(70, 90, 95)) -> dict:
    """Multi-quality ELA: tampered regions often respond differently across
    recompression qualities than authentic regions, so the *change* in ELA
    response across qualities (not just its value at one quality) carries
    extra signal. We compute ELA at each quality plus a delta between the
    lowest and highest quality response."""
    feats = {}
    ela_maps = {}
    for q in qualities:
        ela_map = compute_ela(path, quality=q)
        ela_maps[q] = ela_map
        feats.update(ela_summary_features(ela_map, prefix=f"ela_q{q}"))

    # Cross-quality delta: how much the ELA response changes between the
    # most aggressive and least aggressive recompression
    q_low, q_high = min(qualities), max(qualities)
    delta = np.abs(ela_maps[q_low].astype(np.int16) - ela_maps[q_high].astype(np.int16)).astype(np.uint8)
    feats.update(ela_summary_features(delta, prefix="ela_delta"))

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    feats.update(noise_features(gray))
    feats.update(edge_features(gray))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, default=Path("data/master_index.csv"))
    ap.add_argument("--out", type=Path, default=Path("features/ela_features.csv"))
    ap.add_argument("--qualities", type=int, nargs="+", default=[70, 90, 95],
                     help="JPEG qualities to compute multi-scale ELA at")
    args = ap.parse_args()

    df = pd.read_csv(args.master_csv)
    rows = []
    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ELA/noise features"):
        try:
            feats = extract_all_features(row["image_path"], qualities=tuple(args.qualities))
            feats["case_id"] = row["case_id"]
            rows.append(feats)
        except Exception as e:
            failed.append((row["case_id"], str(e)))

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    if failed:
        print(f"{len(failed)} images failed, e.g.: {failed[:5]}")


if __name__ == "__main__":
    main()
