"""
faiss_manager.py

FAISS manager for Identix.

IMPORTANT:
    Each student can have MANY face embeddings.

Example:

    Student A -> 127 embeddings
    Student B -> 145 embeddings
    Student C -> 132 embeddings

Every embedding is stored individually in FAISS.

FAISS index:
    IndexFlatIP

Because all embeddings are L2-normalized,
inner product is equivalent to cosine similarity.
"""

from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np

from app.config.settings import FAISS_MODEL_DIR


class FAISSManager:
    """
    Manages the FAISS vector database for Identix.

    One student can have multiple embeddings.
    """

    def __init__(self):

        # ==================================================
        # Configuration
        # ==================================================

        self.dimension = 512

        # --------------------------------------------------
        # FAISS index file
        # --------------------------------------------------

        self.index_path = (
            FAISS_MODEL_DIR
            / "student.index"
        )

        # --------------------------------------------------
        # Mapping file
        #
        # Position in FAISS
        #        ↓
        # register number
        # --------------------------------------------------

        self.id_path = (
            FAISS_MODEL_DIR
            / "student_ids.npy"
        )

        # --------------------------------------------------
        # Make sure directory exists
        # --------------------------------------------------

        FAISS_MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==================================================
        # Load Existing Index
        # ==================================================

        if self.index_path.exists():

            self.index = faiss.read_index(
                str(self.index_path)
            )

            # ----------------------------------------------
            # Load mapping
            # ----------------------------------------------

            if self.id_path.exists():

                self.student_ids = (
                    np.load(
                        self.id_path,
                        allow_pickle=True
                    ).tolist()
                )

            else:

                self.student_ids = []

            # ----------------------------------------------
            # Validate mapping
            # ----------------------------------------------

            if (
                self.index.ntotal
                != len(self.student_ids)
            ):

                raise ValueError(
                    "FAISS index and student ID "
                    "mapping have different sizes."
                )

        # ==================================================
        # Create New Index
        # ==================================================

        else:

            self.index = faiss.IndexFlatIP(
                self.dimension
            )

            self.student_ids = []

    # ======================================================
    # Save Index
    # ======================================================

    def save(self):
        """
        Save FAISS index and student mapping.
        """

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        np.save(
            self.id_path,
            np.array(
                self.student_ids,
                dtype=str
            )
        )

    # ======================================================
    # Validate Embedding
    # ======================================================

    def _prepare_embedding(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize one embedding.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Flatten
        # --------------------------------------------------

        embedding = embedding.reshape(-1)

        # --------------------------------------------------
        # Validate dimension
        # --------------------------------------------------

        if embedding.shape[0] != self.dimension:

            raise ValueError(
                f"Invalid embedding dimension. "
                f"Expected {self.dimension}, "
                f"got {embedding.shape[0]}."
            )

        # --------------------------------------------------
        # Check NaN / Inf
        # --------------------------------------------------

        if not np.isfinite(
            embedding
        ).all():

            raise ValueError(
                "Embedding contains NaN or Inf values."
            )

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:

            raise ValueError(
                "Cannot normalize zero embedding."
            )

        embedding = (
            embedding / norm
        ).astype(
            np.float32
        )

        return embedding

    # ======================================================
    # Add One Embedding
    # ======================================================

    def add_embedding(
        self,
        register_number: str,
        embedding: np.ndarray,
    ):
        """
        Add ONE face embedding for a student.

        Example:

            Student 003
                ↓
            embedding 1

        Another call:

            Student 003
                ↓
            embedding 2

        Both are stored separately.
        """

        embedding = self._prepare_embedding(
            embedding
        )

        # --------------------------------------------------
        # Add to FAISS
        # --------------------------------------------------

        self.index.add(
            embedding.reshape(
                1,
                -1
            )
        )

        # --------------------------------------------------
        # Store mapping
        # --------------------------------------------------

        self.student_ids.append(
            str(register_number)
        )

    # ======================================================
    # Add Multiple Embeddings
    # ======================================================

    def add_embeddings(
        self,
        register_number: str,
        embeddings: np.ndarray,
        save: bool = True,
    ):
        """
        Add multiple embeddings belonging to
        one student.

        embeddings shape:

            (number_of_faces, 512)
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Validate shape
        # --------------------------------------------------

        if embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:

            raise ValueError(
                f"Invalid embedding dimension. "
                f"Expected {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if embeddings.shape[0] == 0:

            return 0

        # --------------------------------------------------
        # Normalize every embedding
        # --------------------------------------------------

        embeddings = (
            embeddings.copy()
            .astype(np.float32)
        )

        faiss.normalize_L2(
            embeddings
        )

        # --------------------------------------------------
        # Validate values
        # --------------------------------------------------

        if not np.isfinite(
            embeddings
        ).all():

            raise ValueError(
                "Embeddings contain NaN or Inf values."
            )

        # --------------------------------------------------
        # Add all vectors at once
        # --------------------------------------------------

        self.index.add(
            embeddings
        )

        # --------------------------------------------------
        # Add student mapping for every vector
        # --------------------------------------------------

        self.student_ids.extend(
            [
                str(register_number)
                for _ in range(
                    embeddings.shape[0]
                )
            ]
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        if save:

            self.save()

        return embeddings.shape[0]

    # ======================================================
    # Add Student From NPY File
    # ======================================================

    def add_student_from_file(
        self,
        register_number: str,
        embedding_file: Path,
        save: bool = True,
    ):
        """
        Load a student's .npy embedding file
        and add every embedding to FAISS.

        Example:

            727723euai003.npy

        shape:

            (127, 512)
        """

        embedding_file = Path(
            embedding_file
        )

        if not embedding_file.exists():

            raise FileNotFoundError(
                f"Embedding file not found:\n"
                f"{embedding_file}"
            )

        embeddings = np.load(
            embedding_file
        )

        added = self.add_embeddings(
            register_number,
            embeddings,
            save=False
        )

        if save:

            self.save()

        return added

    # ======================================================
    # Search
    # ======================================================

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
    ):
        """
        Search for the nearest face embeddings.

        Returns individual embedding matches.

        Example:

            [
                {
                    "register_number":
                        "727723euai003",
                    "similarity":
                        0.91,
                    "faiss_index":
                        42
                },
                ...
            ]
        """

        if self.index.ntotal == 0:

            return []

        # --------------------------------------------------
        # Validate top_k
        # --------------------------------------------------

        top_k = max(
            1,
            min(
                top_k,
                self.index.ntotal
            )
        )

        # --------------------------------------------------
        # Prepare query
        # --------------------------------------------------

        embedding = self._prepare_embedding(
            embedding
        )

        query = embedding.reshape(
            1,
            -1
        )

        # --------------------------------------------------
        # FAISS search
        # --------------------------------------------------

        distances, indices = (
            self.index.search(
                query,
                top_k
            )
        )

        results = []

        # --------------------------------------------------
        # Convert results
        # --------------------------------------------------

        for score, index in zip(
            distances[0],
            indices[0]
        ):

            if index == -1:

                continue

            register_number = (
                self.student_ids[index]
            )

            results.append(
                {
                    "register_number":
                        register_number,

                    "similarity":
                        float(score),

                    "faiss_index":
                        int(index),
                }
            )

        return results

    # ======================================================
    # Student-Level Search
    # ======================================================

    def search_students(
        self,
        embedding: np.ndarray,
        top_k: int = 20,
        student_limit: int = 5,
    ):
        """
        Search multiple face embeddings and aggregate
        the results by student.

        This is important because one student has
        multiple embeddings.

        Example:

            Student A:
                0.91
                0.89
                0.87

            Student B:
                0.84
                0.82

        Student A receives stronger evidence.

        Returns students ranked by their strongest
        matching embeddings.
        """

        results = self.search(
            embedding,
            top_k=top_k
        )

        if not results:

            return []

        # --------------------------------------------------
        # Group matches by student
        # --------------------------------------------------

        grouped = defaultdict(list)

        for result in results:

            grouped[
                result["register_number"]
            ].append(
                result["similarity"]
            )

        # --------------------------------------------------
        # Calculate student-level score
        # --------------------------------------------------

        student_results = []

        for register_number, scores in (
            grouped.items()
        ):

            scores = sorted(
                scores,
                reverse=True
            )

            # ----------------------------------------------
            # Strongest match
            # ----------------------------------------------

            best_score = scores[0]

            # ----------------------------------------------
            # Average of top 3 matches
            #
            # This prevents a single accidental match
            # from dominating the result.
            # ----------------------------------------------

            top_scores = scores[:3]

            average_top_score = (
                sum(top_scores)
                / len(top_scores)
            )

            # ----------------------------------------------
            # Combined score
            #
            # We prioritize the strongest face match,
            # while also considering repeated matches.
            # ----------------------------------------------

            final_score = (
                0.70 * best_score
                + 0.30 * average_top_score
            )

            student_results.append(
                {
                    "register_number":
                        register_number,

                    "best_similarity":
                        float(best_score),

                    "average_top_similarity":
                        float(
                            average_top_score
                        ),

                    "final_score":
                        float(final_score),

                    "matching_embeddings":
                        len(scores),
                }
            )

        # --------------------------------------------------
        # Sort by final score
        # --------------------------------------------------

        student_results.sort(
            key=lambda item:
                item["final_score"],
            reverse=True
        )

        # --------------------------------------------------
        # Return top students
        # --------------------------------------------------

        return student_results[
            :student_limit
        ]

    # ======================================================
    # Get Student Count
    # ======================================================

    def get_student_count(self):
        """
        Return number of unique students in FAISS.
        """

        return len(
            set(
                self.student_ids
            )
        )

    # ======================================================
    # Get Embedding Count
    # ======================================================

    def get_embedding_count(self):
        """
        Return total number of embeddings stored.
        """

        return self.index.ntotal

    # ======================================================
    # Get Student Embedding Count
    # ======================================================

    def get_student_embedding_count(
        self,
        register_number: str,
    ):
        """
        Return number of embeddings stored
        for one student.
        """

        return sum(
            1
            for student_id
            in self.student_ids
            if student_id
            == str(register_number)
        )

    # ======================================================
    # Check Student Exists
    # ======================================================

    def has_student(
        self,
        register_number: str,
    ):
        """
        Check whether a student already has
        embeddings stored in FAISS.
        """

        return (
            str(register_number)
            in self.student_ids
        )

    # ======================================================
    # Clear Entire Index
    # ======================================================

    def clear(self):
        """
        Completely clear the FAISS index
        and student mapping.

        USE CAREFULLY.

        This deletes all stored FAISS data.
        """

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.student_ids = []

        self.save()