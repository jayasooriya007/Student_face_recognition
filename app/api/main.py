"""
main.py

FastAPI application entry point.
"""

from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database.database import Base, engine, get_db
from app.database.schema import StudentCreate
from app.services.identify_service import IdentifyService
from app.services.register_service import RegisterService

# ----------------------------------
# Create Database Tables
# ----------------------------------

Base.metadata.create_all(bind=engine)

# ----------------------------------
# FastAPI App
# ----------------------------------

app = FastAPI(
    title="Identix API",
    version="1.0.0",
    description="AI Powered Student Identification System",
)

register_service = RegisterService()
identify_service = IdentifyService()

# ----------------------------------
# Health Check
# ----------------------------------


@app.get("/")
def home():
    return {
        "message": "Welcome to Identix API",
        "status": "Running"
    }


# ----------------------------------
# Register Student
# ----------------------------------


@app.post("/register")
async def register_student(
    student_id: str = Form(...),
    register_number: str = Form(...),
    name: str = Form(...),
    department: str = Form(...),
    year: int = Form(...),
    section: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    video_path = temp_dir / video.filename

    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())

    student = StudentCreate(
        student_id=student_id,
        register_number=register_number,
        name=name,
        department=department,
        year=year,
        section=section,
        phone=phone,
        email=email,
    )

    try:

        result = register_service.register_student(
            db,
            student,
            str(video_path),
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ----------------------------------
# Identify Student
# ----------------------------------


@app.post("/identify")
async def identify_student(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    image_path = temp_dir / image.filename

    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    try:

        result = identify_service.identify_student(
            db,
            str(image_path),
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ----------------------------------
# Get All Students
# ----------------------------------


@app.get("/students")
def get_students(
    db: Session = Depends(get_db),
):

    from app.database.models import Student

    students = db.query(Student).all()

    return students


# ----------------------------------
# Get Student by ID
# ----------------------------------


@app.get("/students/{student_id}")
def get_student(
    student_id: str,
    db: Session = Depends(get_db),
):

    from app.database.models import Student

    student = db.query(Student).filter(
        Student.student_id == student_id
    ).first()

    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    return student


# ----------------------------------
# Delete Student
# ----------------------------------


@app.delete("/students/{student_id}")
def delete_student(
    student_id: str,
    db: Session = Depends(get_db),
):

    from app.database.models import Student

    student = db.query(Student).filter(
        Student.student_id == student_id
    ).first()

    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found.",
        )

    db.delete(student)
    db.commit()

    return {
        "message": "Student deleted successfully."
    }