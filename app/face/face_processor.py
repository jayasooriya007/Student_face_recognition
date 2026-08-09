"""
face_processor.py

SCRFD face detection + ArcFace face alignment.

This file performs ONLY:
    1. Face detection using SCRFD
    2. Face alignment using SCRFD landmarks
    3. Saving aligned 112x112 face images

This file does NOT generate ArcFace embeddings.

Input:
    dataset/quality_filtered/<register_number>/

Output:
    dataset/faces/<register_number>/
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from insightface.utils.face_align import norm_crop

from app.config.settings import (
    QUALITY_FILTERED_DATASET_DIR,
    FACE_DATASET_DIR,
)

from app.face.face_engine import FaceEngine


class FaceProcessor:
    """
    Handles SCRFD detection and ArcFace-compatible
    face alignment.
    """

    def __init__(self):

        self.model = FaceEngine.get_model()

    # ======================================================
    # Detect Faces
    # ======================================================

    def detect_faces(
        self,
        image: np.ndarray,
    ):
        """
        Detect faces using SCRFD.

        The returned Face objects contain:
            - bounding box
            - detection score
            - facial landmarks
        """

        if image is None:

            return []

        return self.model.get(
            image
        )

    # ======================================================
    # Select Best Face
    # ======================================================

    def select_best_face(
        self,
        faces,
    ):
        """
        Select the largest detected face.
        """

        if not faces:

            return None

        best_face = None
        best_area = 0.0

        for face in faces:

            x1, y1, x2, y2 = face.bbox

            width = max(
                0.0,
                x2 - x1
            )

            height = max(
                0.0,
                y2 - y1
            )

            area = width * height

            if area > best_area:

                best_area = area
                best_face = face

        return best_face

    # ======================================================
    # Align Face
    # ======================================================

    def align_face(
        self,
        image: np.ndarray,
        face,
    ) -> Optional[np.ndarray]:
        """
        Align the detected face using its SCRFD
        facial landmarks.

        ArcFace uses a 112x112 aligned face input.
        """

        if image is None:

            return None

        if face is None:

            return None

        landmarks = getattr(
            face,
            "kps",
            None
        )

        if landmarks is None:

            return None

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        # --------------------------------------------------
        # SCRFD should provide five landmarks:
        #
        # left eye
        # right eye
        # nose
        # left mouth
        # right mouth
        # --------------------------------------------------

        if landmarks.shape != (5, 2):

            return None

        aligned_face = norm_crop(
            image,
            landmark=landmarks,
            image_size=112,
        )

        if aligned_face is None:

            return None

        return aligned_face

    # ======================================================
    # Process One Image
    # ======================================================

    def process_image(
        self,
        image_path: Path,
    ) -> Optional[np.ndarray]:
        """
        Read one quality-filtered image,
        detect the face using SCRFD,
        and return an aligned 112x112 face.

        No ArcFace embedding is generated here.
        """

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            return None

        # --------------------------------------------------
        # SCRFD detection
        # --------------------------------------------------

        faces = self.detect_faces(
            image
        )

        if len(faces) == 0:

            return None

        # --------------------------------------------------
        # Select student's face
        # --------------------------------------------------

        best_face = self.select_best_face(
            faces
        )

        if best_face is None:

            return None

        # --------------------------------------------------
        # Align face using landmarks
        # --------------------------------------------------

        aligned_face = self.align_face(
            image,
            best_face
        )

        if aligned_face is None:

            return None

        return aligned_face

    # ======================================================
    # Process Student
    # ======================================================

    def process_student(
        self,
        register_number: str,
    ):
        """
        Process all quality-filtered images
        for one student.

        Input:
            dataset/quality_filtered/<register_number>/

        Output:
            dataset/faces/<register_number>/

        Every output image is:
            112 x 112
            ArcFace-aligned
        """

        input_folder = (
            QUALITY_FILTERED_DATASET_DIR
            / register_number
        )

        output_folder = (
            FACE_DATASET_DIR
            / register_number
        )

        if not input_folder.exists():

            raise FileNotFoundError(
                f"Quality-filtered folder not found:\n"
                f"{input_folder}"
            )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        total_images = 0
        aligned_faces = 0
        rejected_images = 0

        # --------------------------------------------------
        # Process every quality-filtered frame
        # --------------------------------------------------

        for image_path in sorted(
            input_folder.iterdir()
        ):

            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                ]
            ):

                continue

            total_images += 1

            aligned_face = (
                self.process_image(
                    image_path
                )
            )

            # --------------------------------------------------
            # Alignment failed
            # --------------------------------------------------

            if aligned_face is None:

                rejected_images += 1

                continue

            # --------------------------------------------------
            # Save aligned face
            # --------------------------------------------------

            filename = (
                f"face_{aligned_faces:04d}.jpg"
            )

            output_path = (
                output_folder
                / filename
            )

            success = cv2.imwrite(
                str(output_path),
                aligned_face
            )

            if not success:

                raise IOError(
                    f"Unable to save aligned face:\n"
                    f"{output_path}"
                )

            aligned_faces += 1

        return {
            "register_number": register_number,
            "total_images": total_images,
            "aligned_faces": aligned_faces,
            "rejected_images": rejected_images,
            "output_folder": str(
                output_folder
            ),
        }