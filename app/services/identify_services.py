"""
identify_service.py

Handles student identification using
ArcFace + FAISS + PostgreSQL.
"""

from sqlalchemy.orm import Session

from app.database.models import Student
from app.face.faiss_manager import FAISSManager
from app.face.recognizer import FaceRecognizer


class IdentifyService:

    def __init__(self):
        self.recognizer = FaceRecognizer()
        self.faiss = FAISSManager()

    def identify_student(
        self,
        db: Session,
        image_path: str,
        threshold: float = 0.50,
    ):
        """
        Identify a student from an uploaded image.

        Workflow:
        1. Generate embedding.
        2. Search FAISS.
        3. Verify similarity.
        4. Fetch student details.
        """

        # ----------------------------------
        # Generate Embedding
        # ----------------------------------

        embedding = self.recognizer.get_embedding_from_path(
            image_path
        )

        if embedding is None:
            return {
                "success": False,
                "message": "No face detected."
            }

        # ----------------------------------
        # Search FAISS
        # ----------------------------------

        results = self.faiss.search(
            embedding,
            top_k=1
        )

        if not results:
            return {
                "success": False,
                "message": "No registered students found."
            }

        best_match = results[0]

        # ----------------------------------
        # Similarity Check
        # ----------------------------------

        if best_match["similarity"] < threshold:
            return {
                "success": False,
                "message": "Student not recognized.",
                "similarity": round(
                    best_match["similarity"],
                    4
                )
            }

        # ----------------------------------
        # Retrieve Student Details
        # ----------------------------------

        student = db.query(Student).filter(
            Student.student_id ==
            best_match["student_id"]
        ).first()

        if student is None:
            return {
                "success": False,
                "message": "Student record not found."
            }

        # ----------------------------------
        # Success Response
        # ----------------------------------

        return {

            "success": True,

            "message": "Student identified successfully.",

            "similarity": round(
                best_match["similarity"],
                4
            ),

            "student": {

                "student_id": student.student_id,

                "register_number":
                student.register_number,

                "name":
                student.name,

                "department":
                student.department,

                "year":
                student.year,

                "section":
                student.section,

                "phone":
                student.phone,

                "email":
                student.email,

                "dataset_path":
                student.dataset_path,

                "video_path":
                student.video_path,
            }
        }