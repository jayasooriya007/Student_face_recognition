"""
bulk_registration_service.py

Handles bulk processing of the existing student dataset.

Current workflow:

Student_dataset/
        |
        |-- Student folders
        |
        |-- Registration videos
                |
                |-- Already extracted
                        |
                        v
                dataset/processed/
                        |
                        v
                Quality Filtering
                        |
                        v
                dataset/quality_filtered/
"""

from pathlib import Path

import cv2

from app.config.settings import (
    PROCESSED_DATASET_DIR,
    QUALITY_FILTERED_DATASET_DIR,
    STUDENT_DATASET_DIR,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

from app.utils.quality_utils import is_good_quality


class BulkRegistrationService:
    """
    Handles bulk processing of the student dataset.
    """

    def __init__(self):

        self.dataset_path = STUDENT_DATASET_DIR

    # ======================================================
    # Get Student Folders
    # ======================================================

    def get_student_folders(self):
        """
        Returns all student folders from the original
        Student_dataset directory.
        """

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found:\n"
                f"{self.dataset_path}"
            )

        folders = [
            folder
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        ]

        folders.sort()

        return folders

    # ======================================================
    # Find Registration Video
    # ======================================================

    def find_registration_video(
        self,
        student_folder: Path,
    ):
        """
        Finds the registration video inside a student's
        original dataset folder.

        This function is only used to verify that the
        student has a video.

        It does NOT extract frames.
        """

        for file in student_folder.iterdir():

            if (
                file.is_file()
                and file.suffix.lower()
                in VIDEO_EXTENSIONS
            ):

                return file

        return None

    # ======================================================
    # Get Existing Processed Frames
    # ======================================================

    def get_processed_frames(
        self,
        register_number: str,
    ):
        """
        Gets the frames that were already extracted
        from the student's video.

        IMPORTANT:
        This function does NOT extract frames again.
        """

        processed_folder = (
            PROCESSED_DATASET_DIR
            / register_number
        )

        if not processed_folder.exists():

            return []

        images = []

        for file in processed_folder.iterdir():

            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                images.append(file)

        images.sort()

        return images

    # ======================================================
    # Quality Filter Existing Frames
    # ======================================================

    def filter_student_frames(
        self,
        register_number: str,
    ):
        """
        Reads the already extracted frames from:

        dataset/processed/<register_number>/

        and copies only good-quality frames to:

        dataset/quality_filtered/<register_number>/

        The original processed frames are NOT modified.
        """

        processed_folder = (
            PROCESSED_DATASET_DIR
            / register_number
        )

        quality_folder = (
            QUALITY_FILTERED_DATASET_DIR
            / register_number
        )

        if not processed_folder.exists():

            raise FileNotFoundError(
                f"Processed frames not found:\n"
                f"{processed_folder}"
            )

        quality_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        frames = self.get_processed_frames(
            register_number
        )

        total_frames = 0
        good_frames = 0
        rejected_frames = 0

        for image_path in frames:

            total_frames += 1

            image = cv2.imread(
                str(image_path)
            )

            # ------------------------------------------
            # Invalid Image
            # ------------------------------------------

            if image is None:

                rejected_frames += 1

                continue

            # ------------------------------------------
            # Quality Check
            # ------------------------------------------

            if not is_good_quality(image):

                rejected_frames += 1

                continue

            # ------------------------------------------
            # Save Good Frame
            # ------------------------------------------

            output_filename = (
                f"good_{good_frames:04d}.jpg"
            )

            output_path = (
                quality_folder
                / output_filename
            )

            success = cv2.imwrite(
                str(output_path),
                image,
            )

            if not success:

                raise IOError(
                    f"Unable to save image:\n"
                    f"{output_path}"
                )

            good_frames += 1

        return {
            "total_frames": total_frames,
            "good_frames": good_frames,
            "rejected_frames": rejected_frames,
            "output_folder": quality_folder,
        }

    # ======================================================
    # Process One Student
    # ======================================================

    def process_student(
        self,
        student_folder: Path,
    ):
        """
        Processes one student using the frames that
        have ALREADY been extracted.

        Workflow:

        Existing processed frames
                |
                v
        Quality filtering
                |
                v
        Good frames
        """

        register_number = student_folder.name

        # ------------------------------------------
        # Check Original Video
        # ------------------------------------------

        video_path = (
            self.find_registration_video(
                student_folder
            )
        )

        if video_path is None:

            return {
                "register_number": register_number,
                "status": "NO_VIDEO",
                "video": None,
            }

        # ------------------------------------------
        # Check Existing Frames
        # ------------------------------------------

        processed_frames = (
            self.get_processed_frames(
                register_number
            )
        )

        if len(processed_frames) == 0:

            return {
                "register_number": register_number,
                "status": "NO_PROCESSED_FRAMES",
                "video": str(video_path),
            }

        # ------------------------------------------
        # Quality Filtering
        # ------------------------------------------

        quality_result = (
            self.filter_student_frames(
                register_number
            )
        )

        return {
            "register_number": register_number,
            "status": "SUCCESS",
            "video": str(video_path),
            "processed_folder": str(
                PROCESSED_DATASET_DIR
                / register_number
            ),
            "total_frames_checked": (
                quality_result[
                    "total_frames"
                ]
            ),
            "good_frames": (
                quality_result[
                    "good_frames"
                ]
            ),
            "rejected_frames": (
                quality_result[
                    "rejected_frames"
                ]
            ),
            "quality_filtered_folder": str(
                quality_result[
                    "output_folder"
                ]
            ),
        }