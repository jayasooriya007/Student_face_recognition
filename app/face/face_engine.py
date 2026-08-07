"""
face_engine.py

Loads the InsightFace FaceAnalysis model only once.
SCRFD is used for face detection.
ArcFace is used for face recognition.
"""

from insightface.app import FaceAnalysis


class FaceEngine:
    """
    Singleton Face Engine
    """

    _instance = None

    @classmethod
    def get_model(cls):
        """
        Returns a single FaceAnalysis instance.
        """
        
        if cls._instance is None:

            cls._instance = FaceAnalysis(
                name="buffalo_l"
            )

            cls._instance.prepare(
                ctx_id=0,
                det_size=(640, 640)
            )

        return cls._instance