# CHAIN-X: A Unified Multimodal Framework for Chain-of-Custody Evidence Trust Analysis

Developed as part of the course on "Forensic Analytics" at Vellore Institute of Technology, Chennai.
by "Arista Vania", "Tannya Pasricha"

Subject Code: CSE4074
Semester: Fall 2026-27
Professor: Dr. S. Johanan Joysingh

Traditional image-forensics approaches primarily analyse original digital image files. However, manipulated images can be displayed on a laptop or mobile screen, photographed using another device, compressed, resized, and redistributed. These processes can introduce distortions that weaken or obscure traces of image manipulation.

SCREEN-TRACE addresses this challenge by combining image tampering classification, forgery localization, and evidence survival analysis to understand how forensic evidence changes under different image transformation conditions.

---

## Project Overview

The project focuses on three major tasks:

1. **Image Tampering Classification:** Classifying images into three categories:
   - Authentic
   - Copy-Move Forgery
   - Image Splicing

2. **Forgery Localization:** Identifying and highlighting potentially manipulated regions within an image using pixel-level segmentation.

3. **Evidence Survival Analysis:** Evaluating how effectively image-forensics models retain their detection and localization capabilities after digital degradation and physical screen recapture.

---

## Problem Statement

Image manipulation has become increasingly accessible with modern editing tools and generative technologies. Manipulated images can be circulated as screenshots, photographed from digital displays, or repeatedly compressed before reaching investigators.

Such transformations may alter important forensic characteristics, making conventional image manipulation detection less reliable.

SCREEN-TRACE aims to investigate whether manipulated regions can still be detected and localized after these transformations and quantify the resulting degradation in forensic performance.

---

## System Architecture

```text
              Input Image
                   |
                   v
          Image Preprocessing
                   |
          +--------+--------+
          |                 |
          v                 v
   Tampering             Forgery
 Classification        Localization
   EfficientNet-B0      U-Net Architecture
          |                 |
          v                 v
   Authentic /          Suspicious
   Copy-Move /          Region Mask
   Splicing
          |                 |
          +--------+--------+
                   |
                   v
          Evidence Survival
              Analysis
                   |
          +--------+--------+
          |                 |
          v                 v
   Digital Degradation   Physical Screen
                         Recapture
          |                 |
          +--------+--------+
                   |
                   v
          Comparative Analysis
                   |
                   v
          Gradio Live Dashboard
```

---

## Technologies and Models

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Classification | EfficientNet-B0 |
| Localization | U-Net with lightweight encoder |
| Image Processing | OpenCV, PIL |
| Numerical Processing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn |
| Interactive Interface | Gradio |
| Development Environment | Kaggle Notebooks |

### Forensic Features

The project explores the use of:

- RGB image information
- Error Level Analysis (ELA)
- High-frequency residual features
- Image compression characteristics
- Spatial and visual inconsistencies

These features are evaluated experimentally rather than assumed to improve performance.

---

## Datasets

The project uses public image-forensics datasets and a small controlled screen-recapture collection.

| Dataset | Purpose |
|---|---|
| CASIA v2.0 | Primary image manipulation classification and localization |
| CASIA Ground Truth Masks | Pixel-level forgery localization |
| IMD2020 | External evaluation and generalization analysis |
| SCREEN-TRACE Recapture Dataset | Controlled physical screen-recapture experiments |

Dataset labels, annotation quality, and image-to-mask correspondence are verified before training and evaluation.

### Physical Screen Recapture

A controlled pilot experiment is conducted using images displayed on a laptop or mobile screen and photographed using a camera or smartphone.

Capture conditions such as lighting, viewing angle, camera distance, and device information are recorded.

The recaptured images are compared with their original digital versions to investigate the effect of physical recapture on forensic detection.

---

## Experimental Analysis

### 1. Digital Degradation

Images are subjected to controlled transformations:

- JPEG compression at different quality levels.
- Image resizing and resampling.
- Gaussian blur.
- Repeated compression.
- Sequential image transformations.

### 2. Physical Screen Recapture

Images are displayed on a digital screen and photographed using another device.

This introduces real-world effects such as:

- Perspective distortion.
- Moiré patterns.
- Reflections and glare.
- Blur and resampling.
- Colour and brightness variations.

### 3. Evidence Survival Measurement

The change in forensic performance is measured using:

\[
\Delta M = M_{original} - M_{transformed}
\]

Where:

- \(M_{original}\) represents the metric obtained from the original image.
- \(M_{transformed}\) represents the metric obtained after degradation or recapture.

A positive value indicates a reduction in the measured performance.

---

## Evaluation Metrics

**Classification:**
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- AUROC (where applicable)

**Localization:**
- Intersection over Union (IoU)
- Dice Coefficient
- Pixel Precision
- Pixel Recall

**Evidence Survival:**
- Original versus degraded classification performance.
- Original versus recaptured classification performance.
- Localization performance degradation.
- Comparative analysis across transformation conditions.

---

## Interactive Demonstration

SCREEN-TRACE includes a planned interactive forensic dashboard using Gradio.

The dashboard is designed to allow users to:

1. Upload an image for forensic examination.
2. View the predicted manipulation category.
3. Inspect classification confidence scores.
4. Visualize potentially manipulated regions through segmentation masks.
5. Compare original and degraded or screen-recaptured images.
6. Examine changes in forensic performance.

The dashboard provides an accessible demonstration of the complete forensic analysis pipeline.

---

## Research Contribution

The primary research contribution of SCREEN-TRACE is its investigation of image-forensics reliability under screen-laundering conditions.

Unlike an evaluation limited to original digital image files, this project examines the combined problem of:

- Image manipulation classification.
- Pixel-level forgery localization.
- Digital degradation analysis.
- Physical screen-recapture evaluation.
- Evidence survival measurement.

The project aims to provide an experimental understanding of how forensic detection capabilities change when images undergo realistic display-and-camera transformations.

---

## Repository Structure

```text
SCREEN-TRACE/
│
├── README.md
├── PROJECT_EXPLAINER.md
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_localization.ipynb
│   ├── 04_evidence_survival.ipynb
│   └── 05_change_validation.ipynb
│
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── evaluation/
│   └── inference/
│
├── app/
│   └── app.py
│
├── reports/
│   ├── figures/
│   └── evaluation/
│
├── requirements.txt
└── .gitignore
```

*Note: The structure represents the proposed repository organization and should be updated to match the final implementation.*

---
Experimental Results

Image-Level Detection Results

The image-level detection model was evaluated on authentic and manipulated images to assess its ability to distinguish between the two categories.

Performance Metrics

Metric

Score

ROC-AUC

0.9315

Accuracy

84.6%

Balanced Accuracy

84.6%

Precision

81.68%

Recall

89.2%

F1-score

85.28%

Confusion Matrix

Actual / Predicted

Authentic

Manipulated

Authentic

200

50

Manipulated

27

223

Results Interpretation

ROC-AUC (0.9315): Indicates strong discrimination between authentic and manipulated images across classification thresholds.

Accuracy (84.6%): The model correctly classified 84.6% of the evaluated images.

Precision (81.68%): Of the images predicted as manipulated, 81.68% were actually manipulated.

Recall (89.2%): The model successfully identified 89.2% of the actual manipulated images.

F1-score (85.28%): Demonstrates a balance between precision and recall.

The results indicate that the model identifies manipulated images with relatively high recall, making it useful for preliminary forensic screening. However, false positives and false negatives remain, highlighting the need for further validation and complementary localization analysis.

## Limitations

- The physical screen-recapture collection is exploratory and relatively small.
- Dataset annotations may contain inconsistencies.
- Different cameras, screens, and lighting conditions may affect model performance.
- Model predictions may contain false positives and false negatives.
- A predicted suspicious region does not independently establish that an image has been manipulated.
- Further validation is required before applying the system to real forensic investigations.

SCREEN-TRACE is intended as an academic research prototype and forensic decision-support tool, not a replacement for professional forensic examination.

---

## Project Domain

**Image and Video Analytics | Digital Image Forensics | Deep Learning | Computer Vision**

**SDG Alignment:** SDG 16 – Peace, Justice and Strong Institutions

The project contributes to the broader objective of supporting information integrity and the responsible examination of potentially manipulated digital evidence.

---

