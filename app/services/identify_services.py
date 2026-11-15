"""
identify_service.py

Handles the complete student identification workflow.

Workflow:

    Uploaded Image
        ↓
    SCRFD Face Detection
        ↓
    Select Best Face
        ↓
    Facial Landmark Alignment
        ↓
    112x112 Face
        ↓
    ArcFace Recognition
        ↓
    512-D Query Embedding
        ↓
    FAISS Search
        ↓
    Student-Level Matching
        ↓
    MySQL Student Information

IMPORTANT:

    FAISS is responsible for identifying the closest
    registered face.

    MySQL is responsible for retrieving the student's
    complete information using the register number.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

from insightface.utils.face_align import norm_crop

from app.config.settings import SIMILARITY_THRESHOLD
from app.database.models import Student
from app.face.face_engine import FaceEngine
from app.face.faiss_manager import FAISSManager


class IdentifyService:
    """
    Identifies a student from an uploaded face image.
    """

    def __init__(self):

        # ==================================================
        # Load InsightFace Model
        # ==================================================

        self.model = FaceEngine.get_model()

        # ==================================================
        # Get ArcFace Recognition Model
        # ==================================================

        if "recognition" not in self.model.models:

            raise RuntimeError(
                "ArcFace recognition model "
                "was not loaded."
            )

        self.recognition_model = (
            self.model.models["recognition"]
        )

        # ==================================================
        # FAISS Manager
        # ==================================================

        self.faiss = FAISSManager()

    # ======================================================
    # Load Image
    # ======================================================

    def load_image(
        self,
        image_path: str,
    ) -> np.ndarray:
        """
        Load uploaded image using OpenCV.
        """

        image_path = Path(image_path)

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                f"Unable to read image:\n"
                f"{image_path}"
            )

        return image

    # ======================================================
    # Detect Faces
    # ======================================================

    def detect_faces(
        self,
        image: np.ndarray,
    ):
        """
        Detect faces using SCRFD.
        """

        faces = self.model.get(
            image
        )

        return faces

    # ======================================================
    # Select Best Face
    # ======================================================

    def select_best_face(
        self,
        faces,
    ):
        """
        Select the largest detected face.

        If an uploaded image contains multiple faces,
        the largest face is considered the primary
        identification target.
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
        Align detected face using its five landmarks.

        Output:
            112x112 BGR image suitable for ArcFace.
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
        # Expected:
        #
        # 5 landmarks × 2 coordinates
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

        # --------------------------------------------------
        # Final size validation
        # --------------------------------------------------

        if (
            aligned_face.shape[0] != 112
            or aligned_face.shape[1] != 112
        ):

            aligned_face = cv2.resize(
                aligned_face,
                (112, 112),
                interpolation=cv2.INTER_AREA
            )

        return aligned_face

    # ======================================================
    # Generate Query Embedding
    # ======================================================

    def generate_query_embedding(
        self,
        aligned_face: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Generate a 512-D ArcFace embedding directly
        from an already aligned 112x112 face.

        SCRFD is NOT called here.
        """

        if aligned_face is None:

            return None

        # --------------------------------------------------
        # Ensure 112x112
        # --------------------------------------------------

        if (
            aligned_face.shape[0] != 112
            or aligned_face.shape[1] != 112
        ):

            aligned_face = cv2.resize(
                aligned_face,
                (112, 112),
                interpolation=cv2.INTER_AREA
            )

        # --------------------------------------------------
        # ArcFace Recognition Model
        # --------------------------------------------------

        embedding = (
            self.recognition_model
            .get_feat(aligned_face)
            .flatten()
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Validate
        # --------------------------------------------------

        if embedding.ndim != 1:

            return None

        if embedding.shape[0] != 512:

            raise ValueError(
                "ArcFace returned an unexpected "
                f"embedding dimension: "
                f"{embedding.shape[0]}"
            )

        # --------------------------------------------------
        # Normalize
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
    # Get Student From Database
    # ======================================================

    def get_student(
        self,
        db: Session,
        register_number: str,
    ):
        """
        Retrieve student information from MySQL
        using the register number returned by FAISS.
        """

        # --------------------------------------------------
        # Normalize register number
        # --------------------------------------------------

        register_number = (
            str(register_number)
            .strip()
            .upper()
        )

        student = (
            db.query(Student)
            .filter(
                Student.register_number
                == register_number
            )
            .first()
        )

        return student

    # ======================================================
    # Build Student Response
    # ======================================================

    def build_student_response(
        self,
        student,
    ):
        """
        Convert SQLAlchemy Student object
        into API response data.

        Uses the CURRENT Student model fields.
        """

        return {

            # ----------------------------------------------
            # Basic Student Information
            # ----------------------------------------------

            "register_number":
                student.register_number,

            "student_name":
                student.student_name,

            "department":
                student.department,

            "academic_batch":
                student.academic_batch,

            "year":
                student.year,

            "section":
                student.section,

            # ----------------------------------------------
            # Contact Information
            # ----------------------------------------------

            "college_email":
                student.college_email,

            "personal_email":
                student.personal_email,

            "phone_number":
                student.phone_number,

            # ----------------------------------------------
            # Parent / Guardian Information
            # ----------------------------------------------

            "parent_guardian_name":
                student.parent_guardian_name,

            "parent_guardian_mobile":
                student.parent_guardian_mobile,

            # ----------------------------------------------
            # Mentor Information
            # ----------------------------------------------

            "mentor_name":
                student.mentor_name,

            "mentor_phone":
                student.mentor_phone,

            # ----------------------------------------------
            # Student Status
            # ----------------------------------------------

            "status":
                student.status,

            "passed_out_year":
                student.passed_out_year,
        }

    # ======================================================
    # Identify Student
    # ======================================================

    def identify_student(
        self,
        db: Session,
        image_path: str,
    ):
        """
        Complete student identification process.

        If the uploaded image contains multiple faces, each
        detected face is processed independently.
        """

        # ==================================================
        # 1. Load Image
        # ==================================================

        image = self.load_image(
            image_path
        )

        # ==================================================
        # 2. Detect Faces
        # ==================================================

        faces = self.detect_faces(
            image
        )

        if len(faces) == 0:

            raise ValueError(
                "No face detected in the uploaded image."
            )

        # ==================================================
        # 3. Check FAISS
        # ==================================================

        if (
            self.faiss.get_embedding_count()
            == 0
        ):

            raise ValueError(
                "FAISS index is empty. "
                "No student embeddings are available."
            )

        # ==================================================
        # 4. Single Face
        # ==================================================
        # Keep the existing single-face response exactly as
        # before. Only group photos use the new multi-face
        # response below.

        if len(faces) == 1:

            best_face = faces[0]

            # ==================================================
            # 5. Align Face
            # ==================================================

            aligned_face = self.align_face(
                image,
                best_face
            )

            if aligned_face is None:

                raise ValueError(
                    "Unable to align the detected face."
                )

            # ==================================================
            # 6. Generate ArcFace Embedding
            # ==================================================

            query_embedding = (
                self.generate_query_embedding(
                    aligned_face
                )
            )

            if query_embedding is None:

                raise ValueError(
                    "Unable to generate face embedding."
                )

            # ==================================================
            # 7. Search FAISS
            # ==================================================

            matches = (
                self.faiss.search_students(
                    query_embedding,
                    top_k=20,
                    student_limit=5,
                )
            )

            if not matches:

                return {
                    "identified": False,

                    "message":
                        "No matching student found.",

                    "similarity": 0.0,

                    "best_similarity": 0.0,

                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),
                }

            # ==================================================
            # 8. Get Best Match
            # ==================================================

            best_match = matches[0]

            register_number = (
                best_match[
                    "register_number"
                ]
            )

            final_score = (
                best_match[
                    "final_score"
                ]
            )

            best_similarity = (
                best_match[
                    "best_similarity"
                ]
            )

            matching_embeddings = (
                best_match[
                    "matching_embeddings"
                ]
            )

            # ==================================================
            # 9. Apply Similarity Threshold
            # ==================================================

            if final_score < SIMILARITY_THRESHOLD:

                return {
                    "identified": False,

                    "message":
                        "Face detected, but no "
                        "sufficiently similar student "
                        "was found.",

                    "similarity":
                        float(final_score),

                    "best_similarity":
                        float(best_similarity),

                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),

                    "top_matches":
                        matches,
                }

            # ==================================================
            # 10. Retrieve Student From MySQL
            # ==================================================

            student = self.get_student(
                db,
                register_number
            )

            # ==================================================
            # FAISS Match Exists But Database Record
            # Does Not Exist
            # ==================================================

            if student is None:

                return {
                    "identified": False,

                    "message":
                        "FAISS identified a student, "
                        "but student information was "
                        "not found in the database.",

                    # IMPORTANT:
                    # Keep the register number visible
                    # because FAISS successfully identified it.

                    "register_number":
                        register_number,

                    "similarity":
                        float(final_score),

                    "best_similarity":
                        float(best_similarity),

                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),

                    "top_matches":
                        matches,
                }

            # ==================================================
            # 11. Return Final Result
            # ==================================================

            return {

                "identified": True,

                "message":
                    "Student identified successfully.",

                "register_number":
                    student.register_number,

                "similarity":
                    float(final_score),

                "best_similarity":
                    float(best_similarity),

                "threshold":
                    float(
                        SIMILARITY_THRESHOLD
                    ),

                "matching_embeddings":
                    int(
                        matching_embeddings
                    ),

                "student":
                    self.build_student_response(
                        student
                    ),

                "top_matches":
                    matches,
            }

        # ==================================================
        # 12. Multiple Faces / Group Photo
        # ==================================================

        results = []

        for face_number, face in enumerate(
            faces,
            start=1
        ):

            # ==================================================
            # 13. Align Current Face
            # ==================================================

            aligned_face = self.align_face(
                image,
                face
            )

            if aligned_face is None:

                results.append({
                    "face_number": face_number,
                    "identified": False,
                    "message":
                        "Unable to align the detected face.",
                    "similarity": 0.0,
                    "best_similarity": 0.0,
                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),
                })
                continue

            # ==================================================
            # 14. Generate ArcFace Embedding
            # ==================================================

            query_embedding = (
                self.generate_query_embedding(
                    aligned_face
                )
            )

            if query_embedding is None:

                results.append({
                    "face_number": face_number,
                    "identified": False,
                    "message":
                        "Unable to generate face embedding.",
                    "similarity": 0.0,
                    "best_similarity": 0.0,
                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),
                })
                continue

            # ==================================================
            # 15. Search FAISS
            # ==================================================

            matches = (
                self.faiss.search_students(
                    query_embedding,
                    top_k=20,
                    student_limit=5,
                )
            )

            if not matches:

                results.append({
                    "face_number": face_number,
                    "identified": False,
                    "message":
                        "No matching student found.",
                    "similarity": 0.0,
                    "best_similarity": 0.0,
                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),
                })
                continue

            # ==================================================
            # 16. Get Best Match
            # ==================================================

            best_match = matches[0]

            register_number = (
                best_match[
                    "register_number"
                ]
            )

            final_score = (
                best_match[
                    "final_score"
                ]
            )

            best_similarity = (
                best_match[
                    "best_similarity"
                ]
            )

            matching_embeddings = (
                best_match[
                    "matching_embeddings"
                ]
            )

            # ==================================================
            # 17. Apply Similarity Threshold
            # ==================================================

            if final_score < SIMILARITY_THRESHOLD:

                results.append({
                    "face_number": face_number,
                    "identified": False,
                    "message":
                        "Face detected, but no "
                        "sufficiently similar student "
                        "was found.",
                    "similarity":
                        float(final_score),
                    "best_similarity":
                        float(best_similarity),
                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),
                    "top_matches": matches,
                })
                continue

            # ==================================================
            # 18. Retrieve Student From MySQL
            # ==================================================

            student = self.get_student(
                db,
                register_number
            )

            # ==================================================
            # 19. FAISS Match Exists But Database Record
            # Does Not Exist
            # ==================================================

            if student is None:

                results.append({
                    "face_number": face_number,
                    "identified": False,
                    "message":
                        "FAISS identified a student, "
                        "but student information was "
                        "not found in the database.",
                    "register_number":
                        register_number,
                    "similarity":
                        float(final_score),
                    "best_similarity":
                        float(best_similarity),
                    "threshold":
                        float(
                            SIMILARITY_THRESHOLD
                        ),
                    "top_matches": matches,
                })
                continue

            # ==================================================
            # 20. Add Identified Student Result
            # ==================================================

            results.append({
                "face_number": face_number,
                "identified": True,
                "message":
                    "Student identified successfully.",
                "register_number":
                    student.register_number,
                "similarity":
                    float(final_score),
                "best_similarity":
                    float(best_similarity),
                "threshold":
                    float(
                        SIMILARITY_THRESHOLD
                    ),
                "matching_embeddings":
                    int(
                        matching_embeddings
                    ),
                "student":
                    self.build_student_response(
                        student
                    ),
                "top_matches":
                    matches,
            })

        # ==================================================
        # 21. Return Group Photo Results
        # ==================================================

        identified_count = sum(
            1
            for result in results
            if result.get("identified") is True
        )

        return {
            "identified": identified_count > 0,
            "message":
                f"Processed {len(faces)} faces. "
                f"Identified {identified_count} student(s).",
            "total_faces": len(faces),
            "identified_faces": identified_count,
            "results": results,
        }