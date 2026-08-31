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


def patch_localized_features(ela_map: np.ndarray, gray: np.ndarray, grid_size: int = 8) -> dict:
    """Splits the image into a grid_size x grid_size grid and measures the
    ELA response separately in each cell. Tampered regions are usually
    LOCALIZED (one pasted-in patch), so genuine tampering should show up
    as one or a few patches with an unusually high ELA response compared
    to the rest of the same image — not as a uniform whole-image shift.

    This is a *relative*, self-normalizing signal (each patch is compared
    only to other patches in the SAME image), so unlike multi-quality ELA
    it shouldn't pick up on a dataset's absolute compression settings —
    only on internal inconsistency within a single photo. That makes it
    more likely to generalize across datasets with different JPEG history.
    """
    h, w = ela_map.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    if cell_h == 0 or cell_w == 0:
        # image too small to grid meaningfully; fall back to whole-image stats
        return {
            "patch_ela_mean_of_means": float(np.mean(ela_map)),
            "patch_ela_std_of_means": 0.0,
            "patch_ela_max_zscore": 0.0,
            "patch_ela_max_minus_median": 0.0,
            "patch_edge_std_of_means": 0.0,
        }

    ela_patch_means = []
    edge_patch_means = []
    edges = cv2.Canny(gray, 100, 200)

    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            ela_patch_means.append(float(ela_map[y0:y1, x0:x1].mean()))
            edge_patch_means.append(float(np.mean(edges[y0:y1, x0:x1] > 0)))

    ela_patch_means = np.array(ela_patch_means)
    edge_patch_means = np.array(edge_patch_means)

    mean_of_means = ela_patch_means.mean()
    std_of_means = ela_patch_means.std()
    median = np.median(ela_patch_means)
    # z-score of the single most extreme patch: how many standard deviations
    # does the "hottest" patch sit above the rest of the image?
    max_zscore = float((ela_patch_means.max() - mean_of_means) / (std_of_means + 1e-6))

    return {
        "patch_ela_mean_of_means": float(mean_of_means),
        "patch_ela_std_of_means": float(std_of_means),
        "patch_ela_max_zscore": max_zscore,
        "patch_ela_max_minus_median": float(ela_patch_means.max() - median),
        "patch_edge_std_of_means": float(edge_patch_means.std()),
    }


def extract_all_features(path: str, qualities=(90,), grid_size: int = 8) -> dict:
    """Single-quality ELA (generalizes better, per our earlier ablation) +
    global summary stats + patch-localized anomaly features."""
    feats = {}
    quality = qualities[0] if len(qualities) == 1 else qualities[-1]
    ela_map = compute_ela(path, quality=quality)
    feats.update(ela_summary_features(ela_map, prefix="ela"))

    if len(qualities) > 1:
        # optional: still support multi-quality if explicitly requested,
        # but this is OFF by default now (see main()'s new default)
        ela_maps = {quality: ela_map}
        for q in qualities:
            if q not in ela_maps:
                ela_maps[q] = compute_ela(path, quality=q)
                feats.update(ela_summary_features(ela_maps[q], prefix=f"ela_q{q}"))
        q_low, q_high = min(qualities), max(qualities)
        delta = np.abs(ela_maps[q_low].astype(np.int16) - ela_maps[q_high].astype(np.int16)).astype(np.uint8)
        feats.update(ela_summary_features(delta, prefix="ela_delta"))

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    feats.update(noise_features(gray))
    feats.update(edge_features(gray))
    feats.update(patch_localized_features(ela_map, gray, grid_size=grid_size))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, default=Path("data/master_index.csv"))
    ap.add_argument("--out", type=Path, default=Path("features/ela_features.csv"))
    ap.add_argument("--qualities", type=int, nargs="+", default=[90],
                     help="JPEG quality/qualities for ELA. Default is single-quality "
                          "(90) since multi-quality was found to overfit to a "
                          "dataset's specific compression history in testing.")
    ap.add_argument("--grid-size", type=int, default=8,
                     help="Grid size for patch-localized anomaly features (e.g. 8 = 8x8 = 64 patches)")
    args = ap.parse_args()

    df = pd.read_csv(args.master_csv)
    rows = []
    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ELA/noise/patch features"):
        try:
            feats = extract_all_features(row["image_path"], qualities=tuple(args.qualities), grid_size=args.grid_size)
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
