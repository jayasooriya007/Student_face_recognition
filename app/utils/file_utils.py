"""
file_utils.py

Utility functions for handling files and directories.
"""

import shutil
from pathlib import Path
from uuid import uuid4

from app.config.settings import (
    RAW_DATASET_DIR,
    VIDEO_UPLOAD_DIR,
)


def generate_student_folder(student_id: str) -> Path:
    """
    Creates a folder for storing the student's dataset.

    Example:
    dataset/raw/STU001/
    """

    student_folder = RAW_DATASET_DIR / student_id
    student_folder.mkdir(parents=True, exist_ok=True)

    return student_folder


def generate_video_filename(student_id: str, extension: str) -> str:
    """
    Generates a unique video filename.

    Example:
    STU001_4fa23c.mp4
    """

    unique_id = uuid4().hex[:6]

    return f"{student_id}_{unique_id}{extension}"


def save_uploaded_video(video_path: str, student_id: str) -> Path:
    """
    Copies the uploaded video into uploads/videos/.
    """

    source = Path(video_path)

    if not source.exists():
        raise FileNotFoundError(f"{source} not found.")

    extension = source.suffix

    filename = generate_video_filename(student_id, extension)

    destination = VIDEO_UPLOAD_DIR / filename

    shutil.copy2(source, destination)

    return destination


def get_dataset_path(student_id: str) -> Path:
    """
    Returns the student's dataset folder.
    """

    return RAW_DATASET_DIR / student_id


def delete_video(video_path: Path) -> None:
    """
    Deletes a video if it exists.
    """

    if video_path.exists():
        video_path.unlink()


def delete_student_dataset(student_id: str) -> None:
    """
    Deletes the student's dataset folder.
    """

    dataset = RAW_DATASET_DIR / student_id

    if dataset.exists():
        shutil.rmtree(dataset)