"""Basic image preprocessing shared by ELA and the CNN feature extractor."""
import cv2
import numpy as np


def load_rgb(path: str, target_size=(224, 224)) -> np.ndarray:
    """Load an image, convert to RGB, resize. Returns uint8 HxWx3."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def normalize_for_cnn(img: np.ndarray) -> np.ndarray:
    """Scale to [0,1] float32. Backbone-specific normalization (mean/std)
    is applied inside feature_extractor.py's transform pipeline."""
    return img.astype(np.float32) / 255.0
