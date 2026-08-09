"""
models.py

SQLAlchemy ORM Models for the Identix
AI Powered Student Identification System.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String
)

from app.database.database import Base


class Student(Base):
    """
    Student Database Model
    """

    __tablename__ = "students"

    # ======================================================
    # Primary Key
    # ======================================================

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True
    )

    # ======================================================
    # Student Information
    # ======================================================

    register_number = Column(
        String(20),
        unique=True,
        nullable=False,
        index=True
    )

    student_name = Column(
        String(100),
        nullable=False
    )

    department = Column(
        String(50),
        nullable=False
    )

    academic_batch = Column(
        String(20),
        nullable=False
    )
    # Example:
    # 2023-2027

    year = Column(
        Integer,
        nullable=False
    )

    section = Column(
        String(10),
        nullable=False
    )

    # ======================================================
    # Contact Details
    # ======================================================

    college_email = Column(
        String(120),
        unique=True,
        nullable=False
    )

    personal_email = Column(
        String(120),
        unique=True,
        nullable=True
    )

    phone_number = Column(
        String(15),
        nullable=True
    )

    # ======================================================
    # Mentor Details
    # ======================================================

    mentor_name = Column(
        String(100),
        nullable=True
    )

    mentor_phone = Column(
        String(15),
        nullable=True
    )

    # ======================================================
    # Face Recognition Data
    # ======================================================

    video_path = Column(
        String(500),
        nullable=False
    )

    profile_image_path = Column(
        String(500),
        nullable=True
    )

    # ======================================================
    # Student Status
    # ======================================================

    status = Column(
        String(20),
        nullable=False,
        default="ACTIVE"
    )
    # ACTIVE
    # ALUMNI
    # TRANSFERRED

    passed_out_year = Column(
        Integer,
        nullable=True
    )

    # ======================================================
    # Audit Information
    # ======================================================

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # ======================================================
    # Object Representation
    # ======================================================

    def __repr__(self):

        return (
            f"<Student("
            f"register_number='{self.register_number}', "
            f"student_name='{self.student_name}', "
            f"department='{self.department}', "
            f"academic_batch='{self.academic_batch}', "
            f"year='{self.year}', "
            f"section='{self.section}', "
            f"status='{self.status}')>"
        )