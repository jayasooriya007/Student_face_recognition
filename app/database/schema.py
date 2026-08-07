"""
schema.py

This file defines the Pydantic schemas used for validating
student registration data and API request/response models.
"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class StudentCreate(BaseModel):
    """
    Schema used while registering a new student.
    """

    student_id: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Unique Student ID"
    )

    register_number: str = Field(
        ...,
        min_length=1,
        max_length=30,
        description="College Register Number"
    )

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Student Full Name"
    )

    department: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Department Name"
    )

    year: int = Field(
        ...,
        ge=1,
        le=4,
        description="Academic Year"
    )

    section: str = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Section"
    )

    phone: str = Field(
        ...,
        min_length=10,
        max_length=15,
        description="Phone Number"
    )

    email: EmailStr = Field(
        ...,
        description="Student Email"
    )


class StudentResponse(StudentCreate):
    """
    Response schema returned after successful registration.
    """

    dataset_path: Optional[str] = None
    video_path: Optional[str] = None

    class Config:
        from_attributes = True