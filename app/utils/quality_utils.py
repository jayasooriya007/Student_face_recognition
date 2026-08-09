"""
quality_utils.py

Image quality checks for the Identix face recognition pipeline.

This module checks:
1. Blur
2. Brightness
3. Basic image validity

Poor-quality frames are rejected before face detection.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ==========================================================
# Quality Thresholds
# ==========================================================

# Lower variance means the image is more likely to be blurry.
BLUR_THRESHOLD = 80.0

# Average grayscale brightness range.
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0


# ==========================================================
# Load Image
# ==========================================================

def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image from disk.

    Returns:
        OpenCV image if successful.
        None if the image cannot be read.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    return image


# ==========================================================
# Blur Detection
# ==========================================================

def calculate_blur_score(
    image: np.ndarray,
) -> float:
    """
    Calculate the blur score using
    the variance of the Laplacian.

    Higher score:
        Sharper image

    Lower score:
        More blurry image
    """

    if image is None:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    return float(score)


def is_blurry(
    image: np.ndarray,
    threshold: float = BLUR_THRESHOLD,
) -> bool:
    """
    Determine whether an image is blurry.

    Returns:
        True  -> image is blurry
        False -> image is sufficiently sharp
    """

    score = calculate_blur_score(image)

    return score < threshold


# ==========================================================
# Brightness Detection
# ==========================================================

def calculate_brightness(
    image: np.ndarray,
) -> float:
    """
    Calculate average image brightness.

    Returns:
        Average grayscale intensity from 0 to 255.
    """

    if image is None:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = np.mean(gray)

    return float(brightness)


def is_too_dark(
    image: np.ndarray,
    threshold: float = MIN_BRIGHTNESS,
) -> bool:
    """
    Determine whether an image is too dark.
    """

    brightness = calculate_brightness(image)

    return brightness < threshold


def is_too_bright(
    image: np.ndarray,
    threshold: float = MAX_BRIGHTNESS,
) -> bool:
    """
    Determine whether an image is overexposed
    or excessively bright.
    """

    brightness = calculate_brightness(image)

    return brightness > threshold


# ==========================================================
# Overall Quality Check
# ==========================================================

def is_good_quality(
    image: np.ndarray,
    blur_threshold: float = BLUR_THRESHOLD,
    min_brightness: float = MIN_BRIGHTNESS,
    max_brightness: float = MAX_BRIGHTNESS,
) -> bool:
    """
    Check whether an image is suitable for
    the next stage of the face recognition pipeline.

    Returns:
        True  -> image passes quality checks
        False -> image should be rejected
    """

    if image is None:
        return False

    if is_blurry(
        image,
        blur_threshold,
    ):
        return False

    if is_too_dark(
        image,
        min_brightness,
    ):
        return False

    if is_too_bright(
        image,
        max_brightness,
    ):
        return False

    return True


# ==========================================================
# Get Quality Information
# ==========================================================

def get_quality_info(
    image: np.ndarray,
) -> dict:
    """
    Return quality measurements for an image.

    Useful for debugging and project analysis.
    """

    if image is None:

        return {
            "valid": False,
            "blur_score": 0.0,
            "brightness": 0.0,
            "is_blurry": True,
            "is_too_dark": True,
            "is_too_bright": False,
            "is_good_quality": False,
        }

    blur_score = calculate_blur_score(image)

    brightness = calculate_brightness(image)

    blurry = blur_score < BLUR_THRESHOLD

    too_dark = brightness < MIN_BRIGHTNESS

    too_bright = brightness > MAX_BRIGHTNESS

    good_quality = (
        not blurry
        and not too_dark
        and not too_bright
    )

    return {
        "valid": True,
        "blur_score": blur_score,
        "brightness": brightness,
        "is_blurry": blurry,
        "is_too_dark": too_dark,
        "is_too_bright": too_bright,
        "is_good_quality": good_quality,
    }