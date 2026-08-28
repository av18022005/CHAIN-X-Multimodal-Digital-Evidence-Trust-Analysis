"""
Frozen EfficientNet-B0 embedding extraction, WITH CHECKPOINTING so a dropped
Kaggle/Colab session never loses more than one batch.

Run this ON Kaggle/Colab (GPU). It reads the master CSV, and for every
case_id it doesn't already have a saved embedding for, extracts one and
writes it to features/embeddings/<case_id>.npy immediately, plus appends
the case_id to a "done" manifest so restarts skip completed work.

Usage:
    python -m src.image.feature_extractor --master-csv data/master_index.csv \
        --out-dir features/embeddings --batch-size 100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from tqdm import tqdm

from src.image.preprocess import load_rgb


def get_backbone(device: str):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()  # drop the classification head
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False  # frozen — no training, inference only
    return model, weights.transforms()


def already_done(out_dir: Path, case_id: str) -> bool:
    return (out_dir / f"{case_id}.npy").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", type=Path, default=Path("data/master_index.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("features/embeddings"))
    ap.add_argument("--batch-size", type=int, default=100,
                     help="Checkpoint frequency, not GPU batch size")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess_tf = get_backbone(device)

    df = pd.read_csv(args.master_csv)
    todo = [r for _, r in df.iterrows() if not already_done(args.out_dir, r["case_id"])]
    print(f"{len(df) - len(todo)} already done, {len(todo)} remaining")

    processed_since_checkpoint = 0
    for row in tqdm(todo, desc="EfficientNet-B0 embeddings"):
        try:
            img = load_rgb(row["image_path"], target_size=(224, 224))
            tensor = preprocess_tf(torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor).squeeze(0).cpu().numpy()  # shape (1280,)
            np.save(args.out_dir / f"{row['case_id']}.npy", emb)
        except Exception as e:
            print(f"FAILED {row['case_id']}: {e}")
            continue

        processed_since_checkpoint += 1
        if processed_since_checkpoint >= args.batch_size:
            print(f"[checkpoint] {processed_since_checkpoint} more embeddings saved to disk")
            processed_since_checkpoint = 0

    print("Done. Re-run this script any time — it will skip completed embeddings.")


if __name__ == "__main__":
    main()
