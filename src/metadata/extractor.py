"""
Metadata extraction: pulls EXIF tags and basic file/image properties from
each image. Runs entirely on CPU, no GPU needed.

This does NOT try to fabricate metadata that isn't there — CASIA2 and most
web-sourced tampering datasets strip or never had rich EXIF, so 'no EXIF
found' is itself a real, informative signal (see anomaly.py), not a
failure of this script.
"""
import os
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS, IFD

# EXIF software strings that indicate the image was processed by an editor
# (as opposed to camera firmware, which usually names itself differently,
# e.g. "iOS 15.4" or a camera model string).
KNOWN_EDITORS = [
    "photoshop", "gimp", "lightroom", "snapseed", "picasa",
    "paint.net", "affinity", "pixlr", "canva", "illustrator",
]


def extract_exif(path: str) -> dict:
    """Returns a dict of decoded EXIF tags (empty dict if none present).
    Reads BOTH the base IFD0 (Make, Model, Software, ...) and the Exif
    sub-IFD (DateTimeOriginal, GPSInfo, and most detailed camera tags) —
    PIL's Image.getexif() alone only returns IFD0 and silently misses
    everything in the sub-IFD, which is where DateTimeOriginal lives on
    real photos."""
    try:
        img = Image.open(path)
        raw_exif = img.getexif()
        if not raw_exif:
            return {}
        decoded = {}
        for tag_id, value in raw_exif.items():
            tag = TAGS.get(tag_id, tag_id)
            decoded[tag] = value

        # Pull in the Exif sub-IFD (DateTimeOriginal, etc.)
        try:
            exif_ifd = raw_exif.get_ifd(IFD.Exif)
            for tag_id, value in exif_ifd.items():
                tag = TAGS.get(tag_id, tag_id)
                decoded[tag] = value
        except Exception:
            pass  # no sub-IFD present, that's fine

        # Pull in GPS sub-IFD presence (we only need to know if it exists)
        try:
            gps_ifd = raw_exif.get_ifd(IFD.GPSInfo)
            if gps_ifd:
                decoded["GPSInfo"] = gps_ifd
        except Exception:
            pass

        return decoded
    except Exception:
        return {}


def extract_file_properties(path: str) -> dict:
    """File-system-level properties that don't require EXIF at all."""
    stat = os.stat(path)
    return {
        "file_size_bytes": stat.st_size,
        "fs_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def extract_image_properties(path: str) -> dict:
    """Basic image properties, always available regardless of EXIF."""
    try:
        with Image.open(path) as img:
            return {
                "image_width": img.width,
                "image_height": img.height,
                "image_format": img.format,
                "image_mode": img.mode,
            }
    except Exception:
        return {"image_width": None, "image_height": None, "image_format": None, "image_mode": None}


def extract_all_metadata(path: str) -> dict:
    exif = extract_exif(path)
    file_props = extract_file_properties(path)
    img_props = extract_image_properties(path)

    software = str(exif.get("Software", "")).lower()
    editor_detected = any(editor in software for editor in KNOWN_EDITORS)

    return {
        "exif_present": bool(exif),
        "exif_tag_count": len(exif),
        "camera_make": str(exif.get("Make", "")),
        "camera_model": str(exif.get("Model", "")),
        "software": str(exif.get("Software", "")),
        "editing_software_detected": editor_detected,
        "datetime_original": str(exif.get("DateTimeOriginal", exif.get("DateTime", ""))),
        "gps_present": "GPSInfo" in exif,
        **file_props,
        **img_props,
    }
