"""
image_utils.py

Utility functions for video processing and image extraction.
"""

from pathlib import Path
import cv2


def extract_frames(
    video_path: Path,
    output_folder: Path,
    frame_interval: int = 5
) -> int:
    """
    Extracts frames from a video.

    Args:
        video_path: Path to the input video.
        output_folder: Folder where extracted frames are saved.
        frame_interval: Save one frame every N frames.

    Returns:
        Number of extracted frames.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:

            filename = output_folder / f"frame_{saved_count:04d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved_count += 1

        frame_count += 1

    cap.release()

    return saved_count


def resize_image(image, width=640):
    """
    Resize image while maintaining aspect ratio.
    """

    h, w = image.shape[:2]

    ratio = width / w

    height = int(h * ratio)

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def load_image(image_path: Path):
    """
    Reads an image using OpenCV.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    return image


def save_image(image, output_path: Path):
    """
    Saves an image.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)


def list_images(folder: Path):
    """
    Returns all jpg, jpeg and png images from a folder.
    """

    images = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(folder.glob(extension))

    return sorted(images)