"""
register_service.py

Handles complete student registration workflow.
"""

import numpy as np
from sqlalchemy.orm import Session

from app.database.models import Student
from app.database.schema import StudentCreate
from app.face.faiss_manager import FAISSManager
from app.face.recognizer import FaceRecognizer
from app.utils.file_utils import (
    generate_student_folder,
    save_uploaded_video,
)
from app.utils.image_utils import (
    extract_frames,
    list_images,
)


class RegisterService:

    def __init__(self):
        self.recognizer = FaceRecognizer()
        self.faiss = FAISSManager()

    def register_student(
        self,
        db: Session,
        student: StudentCreate,
        video_path: str,
    ):
        """
        Register a new student.

        Workflow:
        1. Validate duplicate student.
        2. Create dataset folder.
        3. Save uploaded video.
        4. Extract frames.
        5. Generate ArcFace embeddings.
        6. Compute average embedding.
        7. Store embedding in FAISS.
        8. Store student details in PostgreSQL.
        """

        # ----------------------------------
        # Duplicate Validation
        # ----------------------------------

        if db.query(Student).filter(
            Student.student_id == student.student_id
        ).first():
            raise ValueError("Student ID already exists.")

        if db.query(Student).filter(
            Student.register_number == student.register_number
        ).first():
            raise ValueError("Register Number already exists.")

        if db.query(Student).filter(
            Student.email == student.email
        ).first():
            raise ValueError("Email already exists.")

        # ----------------------------------
        # Create Student Dataset Folder
        # ----------------------------------

        dataset_folder = generate_student_folder(
            student.student_id
        )

        # ----------------------------------
        # Save Registration Video
        # ----------------------------------

        saved_video = save_uploaded_video(
            video_path,
            student.student_id,
        )

        # ----------------------------------
        # Extract Frames
        # ----------------------------------

        total_frames = extract_frames(
            saved_video,
            dataset_folder,
        )

        # ----------------------------------
        # Generate Face Embeddings
        # ----------------------------------

        embeddings = []

        images = list_images(dataset_folder)

        for image_path in images:

            embedding = self.recognizer.get_embedding_from_path(
                str(image_path)
            )

            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) == 0:
            raise ValueError(
                "No face detected in the registration video."
            )

        # ----------------------------------
        # Average Embedding
        # ----------------------------------

        embeddings = np.array(
            embeddings,
            dtype=np.float32,
        )

        final_embedding = np.mean(
            embeddings,
            axis=0,
        )

        # Normalize embedding for cosine similarity
        final_embedding /= np.linalg.norm(
            final_embedding
        )

        # ----------------------------------
        # Add Student to FAISS
        # ----------------------------------

        self.faiss.add_student(
            student.student_id,
            final_embedding,
        )

        # ----------------------------------
        # Save Student in Database
        # ----------------------------------

        new_student = Student(
            student_id=student.student_id,
            register_number=student.register_number,
            name=student.name,
            department=student.department,
            year=student.year,
            section=student.section,
            phone=student.phone,
            email=student.email,
            dataset_path=str(dataset_folder),
            video_path=str(saved_video),
        )

        db.add(new_student)
        db.commit()
        db.refresh(new_student)

        # ----------------------------------
        # Response
        # ----------------------------------

        return {
            "message": "Student registered successfully.",
            "student_id": new_student.student_id,
            "frames_extracted": total_frames,
            "embeddings_generated": len(embeddings),
            "dataset_path": str(dataset_folder),
            "video_path": str(saved_video),
        }