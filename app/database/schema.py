"""
schema.py

Pydantic schemas used for
Student Registration and API validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ======================================================
# Student Registration Request
# ======================================================

class StudentCreate(BaseModel):

    register_number: str = Field(
        ...,
        min_length=5,
        max_length=20,
        description="College Register Number"
    )

    student_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Student Name with Initial"
    )

    department: str = Field(
        ...,
        min_length=2,
        max_length=50
    )

    academic_batch: str = Field(
        ...,
        description="Example: 2023-2027"
    )

    year: int = Field(
        ...,
        ge=1,
        le=4
    )

    section: str = Field(
        ...,
        min_length=1,
        max_length=10
    )

    college_email: EmailStr

    personal_email: Optional[EmailStr] = None

    phone_number: Optional[str] = Field(
        default=None,
        max_length=15
    )

    mentor_name: Optional[str] = None

    mentor_phone: Optional[str] = None

    video_path: str

    profile_image_path: Optional[str] = None

    status: str = "ACTIVE"

    passed_out_year: Optional[int] = None


# ======================================================
# Student Response
# ======================================================

class StudentResponse(StudentCreate):

    id: int

    created_at: datetime

    updated_at: datetime

    class Config:
        from_attributes = True