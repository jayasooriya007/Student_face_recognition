"""
embedding_service.py

Generates individual ArcFace embeddings from
already-aligned 112x112 face images.

IMPORTANT:

SCRFD is NOT used here.

Input:
    dataset/faces/<register_number>/

Output:
    dataset/embeddings/<register_number>.npy

    dataset/embeddings/<register_number>_images.npy
"""

from pathlib import Path

import cv2
import numpy as np

from app.config.settings import (
    EMBEDDINGS_DIR,
)

from app.face.face_engine import FaceEngine


class EmbeddingService:
    """
    Generates individual ArcFace embeddings
    from already-aligned face images.
    """

    def __init__(self):

        self.model = FaceEngine.get_model()

        # --------------------------------------------------
        # Get ArcFace recognition model directly.
        #
        # FaceAnalysis stores models by task name.
        # --------------------------------------------------

        if "recognition" not in self.model.models:

            raise RuntimeError(
                "ArcFace recognition model was not loaded."
            )

        self.recognition_model = (
            self.model.models["recognition"]
        )

    # ======================================================
    # Get Face Images
    # ======================================================

    def get_face_images(
        self,
        faces_folder: Path,
    ):
        """
        Return all aligned face images.
        """

        if not faces_folder.exists():

            return []

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
        }

        images = []

        for image_path in faces_folder.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in extensions
            ):

                images.append(image_path)

        images.sort()

        return images

    # ======================================================
    # Generate ArcFace Embedding
    # ======================================================

    def generate_embedding(
        self,
        image_path: Path,
    ):
        """
        Generate one ArcFace embedding.

        IMPORTANT:

        This function does NOT call:

            self.model.get()

        Therefore SCRFD is NOT executed.

        It directly calls the ArcFace recognition
        model on the already-aligned face.
        """

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            return None

        # --------------------------------------------------
        # Validate expected ArcFace input size
        # --------------------------------------------------

        if (
            image.shape[0] != 112
            or image.shape[1] != 112
        ):

            image = cv2.resize(
                image,
                (112, 112),
                interpolation=cv2.INTER_AREA
            )

        # --------------------------------------------------
        # Direct ArcFace inference
        # --------------------------------------------------

        embedding = (
            self.recognition_model
            .get_feat(image)
            .flatten()
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Validate embedding
        # --------------------------------------------------

        if embedding.ndim != 1:

            return None

        if embedding.shape[0] != 512:

            return None

        # --------------------------------------------------
        # L2 normalization
        # --------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:

            return None

        embedding = (
            embedding / norm
        ).astype(
            np.float32
        )

        return embedding

    # ======================================================
    # Generate Student Embeddings
    # ======================================================

    def generate_student_embeddings(
        self,
        register_number: str,
        faces_folder: Path,
    ):
        """
        Generate one 512-D ArcFace embedding
        for every aligned face image.

        NO averaging is performed.
        """

        face_images = self.get_face_images(
            faces_folder
        )

        if not face_images:

            raise ValueError(
                f"No aligned face images found for "
                f"{register_number}"
            )

        embeddings = []
        image_names = []

        failed = 0

        # --------------------------------------------------
        # Process every aligned face
        # --------------------------------------------------

        for image_path in face_images:

            embedding = self.generate_embedding(
                image_path
            )

            if embedding is None:

                failed += 1

                continue

            embeddings.append(
                embedding
            )

            image_names.append(
                image_path.name
            )

        if not embeddings:

            raise ValueError(
                f"No valid ArcFace embeddings generated "
                f"for {register_number}"
            )

        # --------------------------------------------------
        # Convert to matrix
        # --------------------------------------------------

        embeddings_array = np.vstack(
            embeddings
        ).astype(
            np.float32
        )

        # --------------------------------------------------
        # Create directory
        # --------------------------------------------------

        EMBEDDINGS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # Save embeddings
        # --------------------------------------------------

        embedding_file = (
            EMBEDDINGS_DIR
            / f"{register_number}.npy"
        )

        np.save(
            embedding_file,
            embeddings_array
        )

        # --------------------------------------------------
        # Save mapping
        # --------------------------------------------------

        image_names_file = (
            EMBEDDINGS_DIR
            / f"{register_number}_images.npy"
        )

        np.save(
            image_names_file,
            np.array(
                image_names,
                dtype=str
            )
        )

        return {
            "register_number": register_number,
            "faces_found": len(face_images),
            "embeddings_generated": len(
                embeddings
            ),
            "embeddings_failed": failed,
            "embedding_dimension": 512,
            "embedding_file": str(
                embedding_file
            ),
            "image_names_file": str(
                image_names_file
            ),
        }