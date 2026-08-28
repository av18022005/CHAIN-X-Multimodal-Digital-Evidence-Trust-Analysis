# Dataset acquisition (do this manually — do not let any tool "invent" this data)

## 1. CASIA 2.0 (image tampering, main dataset)

- Images: search "CASIA2 dataset download" — it's mirrored on Kaggle
  ("CASIA 2.0 Image Tampering Detection Dataset") and referenced from
  https://github.com/namtpham/casia2groundtruth (ground truth masks).
- Kaggle is the easiest route since you're already using Kaggle notebooks:
  add the CASIA2 Kaggle dataset directly as a Kaggle "Add Data" source —
  no local download needed.
- You only need a SUBSET. After downloading/mounting, run
  `scripts/build_master_csv.py --sample` (below) to randomly select:
  - 500 authentic
  - 500 spliced
  - 300 copy-move
- Place (or symlink, on Kaggle just point the config path) so the folder
  looks like:
  ```
  data/images/authentic/*.jpg
  data/images/spliced/*.jpg
  data/images/copy_move/*.jpg
  ```

## 2. COVERAGE (external test set — never used in training)

- Search "COVERAGE copy-move forgery dataset" (Wen et al.). Also mirrored
  on GitHub/Kaggle.
- Take ~200-300 images.
- Place in:
  ```
  data/external/coverage/authentic/*.tif
  data/external/coverage/tampered/*.tif
  ```
- Do NOT move any of these into `data/images/`. The classifier script
  refuses to train if it detects overlap between the two folders (see
  `src/image/classifier.py::assert_no_leakage`).

## 3. NIST CFReDS (provenance) — for Phase 6, not needed yet

- Browse https://cfreds.nist.gov and pick ONE small, well-documented
  scenario (e.g. a small disk/mobile image with a clear timeline). Do not
  download large multi-GB images. Note the license/usage terms on the page.

## 4. Dundee Digital Forensics Corpus (NLP) — for Phase 4, not needed yet

- Available via University of Dundee Discovery Portal (CC BY). Download
  the text corpus, not needed until Phase 4.

---

Once CASIA2 + COVERAGE are on disk (or mounted in Kaggle), run:

```bash
python scripts/build_master_csv.py \
    --casia-authentic /path/to/casia/Au \
    --casia-tampered /path/to/casia/Tp \
    --coverage-authentic /path/to/coverage/authentic \
    --coverage-tampered /path/to/coverage/tampered \
    --sample
```

This produces `data/master_index.csv`, the single file every later phase
reads from.
