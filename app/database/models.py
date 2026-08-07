"""
models.py

SQLAlchemy ORM models for the Identix project.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from app.database.database import Base


class Student(Base):
    """
    Student Database Model
    """

    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )

    student_id = Column(
        String(20),
        unique=True,
        nullable=False,
        index=True
    )

    register_number = Column(
        String(30),
        unique=True,
        nullable=False
    )

    name = Column(
        String(100),
        nullable=False
    )

    department = Column(
        String(100),
        nullable=False
    )

    year = Column(
        Integer,
        nullable=False
    )

    section = Column(
        String(5),
        nullable=False
    )

    phone = Column(
        String(15),
        nullable=False
    )

    email = Column(
        String(100),
        unique=True,
        nullable=False
    )

    video_path = Column(
        String(255),
        nullable=True
    )

    dataset_path = Column(
        String(255),
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    def __repr__(self):
        return (
            f"<Student("
            f"student_id='{self.student_id}', "
            f"name='{self.name}', "
            f"department='{self.department}')>"
        )