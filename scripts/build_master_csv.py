"""
Build data/master_index.csv — the single spine file every CHAIN-X module
reads from and writes results into.

Usage:
    python scripts/build_master_csv.py \
        --casia-authentic /path/to/CASIA2/Au \
        --casia-tampered  /path/to/CASIA2/Tp \
        --coverage-authentic /path/to/COVERAGE/authentic \
        --coverage-tampered  /path/to/COVERAGE/tampered \
        --sample

Notes:
    * CASIA2's "Tp" folder mixes spliced and copy-move images; filenames
      contain "_S_" for splicing or "_D_" for copy-move-ish "different" —
      check your actual mirror's naming convention and adjust
      `classify_casia_tamper_type` below before trusting the split.
    * --sample caps CASIA to 500 authentic / 500 spliced / 300 copy-move
      and COVERAGE to whatever is present (it's small already), with a
      fixed random seed so the sample is reproducible.
"""
import argparse
import csv
import hashlib
import os
import random
import sys
from pathlib import Path

SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def classify_casia_tamper_type(path: Path) -> str:
    """Best-effort split of CASIA2 'Tp' filenames into spliced/copy_move.
    Adjust this if your mirror uses different naming (inspect a few
    filenames first — this is exactly the 'inspect before assuming'
    step from the plan)."""
    name = path.stem.upper()
    if "_D_" in name:
        return "copy_move"
    if "_S_" in name:
        return "spliced"
    return "spliced"  # fallback default; verify manually on a sample


def file_hash(path: Path, block_size=65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows(paths_labels, dataset_name, hash_files=True):
    rows = []
    for path, label, tamper_type in paths_labels:
        rows.append(
            {
                "case_id": f"{dataset_name}_{path.stem}",
                "image_path": str(path.resolve()),
                "dataset": dataset_name,
                "label": label,               # authentic / tampered
                "tamper_type": tamper_type,    # spliced / copy_move / none
                "sha256": file_hash(path) if hash_files else "",
                "metadata_available": "",      # filled in Phase 3
                "report_available": "",        # filled in Phase 4
                "provenance_available": "",    # filled in Phase 6
                "split": "",                   # filled below
            }
        )
    return rows


def assign_splits(rows, train=0.7, val=0.15, seed=SEED):
    """Stratified-ish split by label, only for the internal (CASIA) rows.
    COVERAGE rows are always 'external_test'."""
    rnd = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    for label, group in by_label.items():
        rnd.shuffle(group)
        n = len(group)
        n_train = int(n * train)
        n_val = int(n * val)
        for i, r in enumerate(group):
            if i < n_train:
                r["split"] = "train"
            elif i < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "internal_test"


def split_coverage_folder(image_folder: Path):
    """COVERAGE puts everything in one folder: N.tif is the original,
    Nt.tif is the tampered/forged version of the same pair. Split them."""
    if not image_folder or not image_folder.exists():
        return [], []
    all_files = list_images(image_folder)
    authentic, tampered = [], []
    for p in all_files:
        stem = p.stem  # e.g. "1" or "1t"
        if stem.endswith("t"):
            tampered.append((p, "tampered", "copy_move"))
        else:
            authentic.append((p, "authentic", "none"))
    return authentic, tampered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casia-authentic", type=Path)
    ap.add_argument("--casia-tampered", type=Path)
    ap.add_argument("--coverage-authentic", type=Path,
                     help="Use only if your external dataset already has separate authentic/tampered folders")
    ap.add_argument("--coverage-tampered", type=Path)
    ap.add_argument("--coverage-image-dir", type=Path,
                     help="COVERAGE's native layout: one folder with N.tif (original) and Nt.tif (tampered) pairs")
    ap.add_argument("--out", type=Path, default=Path("data/master_index.csv"))
    ap.add_argument("--sample", action="store_true",
                     help="Cap CASIA to the sizes given by --n-authentic/--n-spliced/--n-copymove")
    ap.add_argument("--n-authentic", type=int, default=500,
                     help="Max authentic CASIA images to sample (only used with --sample)")
    ap.add_argument("--n-spliced", type=int, default=500,
                     help="Max spliced CASIA images to sample (only used with --sample)")
    ap.add_argument("--n-copymove", type=int, default=300,
                     help="Max copy-move CASIA images to sample (only used with --sample)")
    ap.add_argument("--no-hash", action="store_true",
                     help="Skip SHA-256 hashing (faster, do this for a quick dry run)")
    args = ap.parse_args()

    rnd = random.Random(SEED)
    all_rows = []

    # ---- CASIA (internal train/val/test) ----
    if args.casia_authentic or args.casia_tampered:
        auth = [(p, "authentic", "none") for p in list_images(args.casia_authentic)] \
            if args.casia_authentic else []
        tamp_paths = list_images(args.casia_tampered) if args.casia_tampered else []
        tamp = [(p, "tampered", classify_casia_tamper_type(p)) for p in tamp_paths]

        if args.sample:
            spliced = [t for t in tamp if t[2] == "spliced"]
            copy_move = [t for t in tamp if t[2] == "copy_move"]
            rnd.shuffle(auth)
            rnd.shuffle(spliced)
            rnd.shuffle(copy_move)
            auth = auth[:args.n_authentic]
            spliced = spliced[:args.n_spliced]
            copy_move = copy_move[:args.n_copymove]
            tamp = spliced + copy_move

        print(f"[CASIA] authentic={len(auth)} spliced="
              f"{sum(1 for t in tamp if t[2]=='spliced')} "
              f"copy_move={sum(1 for t in tamp if t[2]=='copy_move')}")

        casia_rows = build_rows(auth + tamp, "casia", hash_files=not args.no_hash)
        assign_splits(casia_rows)
        all_rows.extend(casia_rows)
    else:
        print("[CASIA] no paths given, skipping", file=sys.stderr)

    # ---- COVERAGE (external test only, never trained on) ----
    if args.coverage_image_dir:
        cov_auth, cov_tamp = split_coverage_folder(args.coverage_image_dir)
        print(f"[COVERAGE, native layout] authentic={len(cov_auth)} tampered={len(cov_tamp)}")
    elif args.coverage_authentic or args.coverage_tampered:
        cov_auth = [(p, "authentic", "none") for p in list_images(args.coverage_authentic)] \
            if args.coverage_authentic else []
        cov_tamp = [(p, "tampered", "copy_move") for p in list_images(args.coverage_tampered)] \
            if args.coverage_tampered else []
        print(f"[COVERAGE, split layout] authentic={len(cov_auth)} tampered={len(cov_tamp)}")
    else:
        cov_auth, cov_tamp = [], []

    if cov_auth or cov_tamp:
        cov_rows = build_rows(cov_auth + cov_tamp, "coverage", hash_files=not args.no_hash)
        for r in cov_rows:
            r["split"] = "external_test"
        all_rows.extend(cov_rows)
    else:
        print("[COVERAGE] no paths given, skipping", file=sys.stderr)

    if not all_rows:
        print("Nothing to write — pass at least one dataset path.", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {args.out}")
    print("Split counts:")
    from collections import Counter
    print(Counter(r["split"] for r in all_rows))


if __name__ == "__main__":
    main()
