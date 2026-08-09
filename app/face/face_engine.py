"""
face_engine.py

Loads the InsightFace FaceAnalysis model only once.

SCRFD  -> Face Detection
ArcFace -> Face Recognition
"""

from insightface.app import FaceAnalysis

from app.config.settings import (
    MODEL_NAME,
    DETECTION_SIZE,
    USE_GPU,
)


class FaceEngine:
    """
    Singleton Face Engine
    """

    _instance = None

    @classmethod
    def get_model(cls):
        """
        Returns a single initialized FaceAnalysis model.
        """

        if cls._instance is None:

            cls._instance = FaceAnalysis(
                name=MODEL_NAME
            )

            cls._instance.prepare(
                ctx_id=0 if USE_GPU else -1,
                det_size=DETECTION_SIZE,
            )

        return cls._instance