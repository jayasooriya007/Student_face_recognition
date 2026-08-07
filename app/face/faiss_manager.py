"""
faiss_manager.py

Handles all FAISS operations.
"""

from pathlib import Path

import faiss
import numpy as np

from app.config.settings import FAISS_MODEL_DIR


class FAISSManager:

    def __init__(self):

        self.dimension = 512

        self.index_path = FAISS_MODEL_DIR / "student.index"

        self.id_path = FAISS_MODEL_DIR / "student_ids.npy"

        if self.index_path.exists():

            self.index = faiss.read_index(str(self.index_path))

            self.student_ids = np.load(
                self.id_path,
                allow_pickle=True
            ).tolist()

        else:

            self.index = faiss.IndexFlatIP(
                self.dimension
            )

            self.student_ids = []

    def save(self):
        """
        Save FAISS index.
        """

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        np.save(
            self.id_path,
            np.array(self.student_ids)
        )

    def add_student(
        self,
        student_id,
        embedding
    ):
        """
        Add new student embedding.
        """

        embedding = embedding.astype(
            np.float32
        )

        faiss.normalize_L2(
            embedding.reshape(1, -1)
        )

        self.index.add(
            embedding.reshape(1, -1)
        )

        self.student_ids.append(
            student_id
        )

        self.save()

    def search(
        self,
        embedding,
        top_k=1
    ):
        """
        Search nearest student.
        """

        if self.index.ntotal == 0:
            return None

        embedding = embedding.astype(
            np.float32
        )

        faiss.normalize_L2(
            embedding.reshape(1, -1)
        )

        distances, indices = self.index.search(
            embedding.reshape(1, -1),
            top_k
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx == -1:
                continue

            results.append({

                "student_id":
                self.student_ids[idx],

                "similarity":
                float(score)

            })

        return results