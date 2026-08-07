"""
detector.py

Face detection module using InsightFace FaceAnalysis.
"""

from pathlib import Path
from typing import List

import cv2

from app.face.face_engine import FaceEngine


class FaceDetector:
    """
    Face Detection using SCRFD via InsightFace.
    """

    def __init__(self):
        self.model = FaceEngine.get_model()

    def detect_faces(self, image):
        """
        Detect all faces in an image.

        Args:
            image: OpenCV image

        Returns:
            List of detected faces
        """

        faces = self.model.get(image)

        return faces

    def detect_from_path(self, image_path: str):
        """
        Detect faces from an image path.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")

        return self.detect_faces(image)

    def crop_face(self, image, face):
        """
        Crop detected face using bounding box.
        """

        x1, y1, x2, y2 = face.bbox.astype(int)

        cropped = image[y1:y2, x1:x2]

        return cropped

    def save_cropped_faces(
        self,
        image_path: str,
        output_folder: str
    ) -> int:
        """
        Detect and save cropped faces.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        faces = self.detect_faces(image)

        output = Path(output_folder)
        output.mkdir(parents=True, exist_ok=True)

        for index, face in enumerate(faces):

            cropped = self.crop_face(image, face)

            cv2.imwrite(
                str(output / f"face_{index}.jpg"),
                cropped
            )

        return len(faces)