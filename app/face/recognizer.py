"""
recognizer.py

Face recognition module using InsightFace FaceAnalysis (ArcFace).
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.face.face_engine import FaceEngine


class FaceRecognizer:
    """
    Face Recognition using ArcFace embeddings.
    """

    def __init__(self):
        self.model = FaceEngine.get_model()

    def get_faces(self, image):
        """
        Detect all faces from an image.

        Returns:
            List of Face objects.
        """
        return self.model.get(image)

    def get_embedding(self, image) -> Optional[np.ndarray]:
        """
        Returns the embedding of the first detected face.
        """

        faces = self.get_faces(image)

        if len(faces) == 0:
            return None

        return faces[0].embedding

    def get_embedding_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generates embedding from an image file.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")

        return self.get_embedding(image)

    def get_all_embeddings(self, folder_path: str):
        """
        Generate embeddings for every image in a folder.
        """

        folder = Path(folder_path)

        embeddings = []
        image_names = []

        for image_path in folder.glob("*"):

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                continue

            embedding = self.get_embedding_from_path(
                str(image_path)
            )

            if embedding is not None:
                embeddings.append(embedding)
                image_names.append(image_path.name)

        return embeddings, image_names

    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Computes cosine similarity between two embeddings.
        """

        similarity = np.dot(
            embedding1,
            embedding2
        ) / (
            np.linalg.norm(embedding1)
            * np.linalg.norm(embedding2)
        )

        return float(similarity)

    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.5
    ):
        """
        Compare two face embeddings.
        """

        similarity = self.cosine_similarity(
            embedding1,
            embedding2
        )

        return similarity >= threshold, similarity