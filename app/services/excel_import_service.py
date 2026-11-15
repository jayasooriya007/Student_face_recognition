"""
excel_import_service.py

Imports student information from the mentor-provided
Excel sheet into the Identix MySQL database.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.database.models import Student


# ==========================================================
# Excel Configuration
# ==========================================================

EXCEL_FILE = Path("data/2023-2027-A.xlsx")
SHEET_NAME = "Full details"

# Mentor information provided for the class
MENTOR_NAME = "Stenni Martin"
MENTOR_PHONE = "9500678584"

# Academic batch
ACADEMIC_BATCH = "2023-2027"


# ==========================================================
# Helper Functions
# ==========================================================

def clean_value(value):
    """
    Convert Excel values into clean strings.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in ["nan", "none", ""]:
        return None

    return value


def clean_phone(value):
    """
    Clean phone numbers that Excel may read as floats.

    Example:
        6369094326.0 -> 6369094326
    """

    value = clean_value(value)

    if value is None:
        return None

    if value.endswith(".0"):
        value = value[:-2]

    return value


def calculate_year():
    """
    Current academic year for the 2023-2027 batch.

    2023-2024 -> Year 1
    2024-2025 -> Year 2
    2025-2026 -> Year 3
    2026-2027 -> Year 4

    Current project date is in the 2026-2027 academic year.
    """

    return 4


# ==========================================================
# Import Function
# ==========================================================

def import_students_from_excel(db: Session):

    if not EXCEL_FILE.exists():

        raise FileNotFoundError(
            f"Excel file not found: {EXCEL_FILE}"
        )

    print("=" * 70)
    print("IDENTIX - EXCEL TO MYSQL IMPORT")
    print("=" * 70)

    # ------------------------------------------------------
    # Read Excel
    # ------------------------------------------------------

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=SHEET_NAME,
        header=4,
    )

    print(f"Excel rows found: {len(df)}")

    # ------------------------------------------------------
    # Required columns
    # ------------------------------------------------------

    required_columns = [
        "Roll no",
        "Name of the student (in caps)",
        "Degree",
        "Programme Code",
        "Section",
        "Email id",
        "Student mobile number",
        "Parent / Guardian Name",
        "Parent / Guardian Mobile Number",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required Excel column missing: {column}"
            )

    # ------------------------------------------------------
    # Import statistics
    # ------------------------------------------------------

    inserted = 0
    updated = 0
    skipped = 0

    year = calculate_year()

    # ------------------------------------------------------
    # Process each student
    # ------------------------------------------------------

    for _, row in df.iterrows():

        register_number = clean_value(
            row["Roll no"]
        )

        student_name = clean_value(
            row["Name of the student (in caps)"]
        )

        department = clean_value(
            row["Programme Code"]
        )

        section = clean_value(
            row["Section"]
        )

        personal_email = clean_value(
            row["Email id"]
        )

        phone_number = clean_phone(
            row["Student mobile number"]
        )

        parent_name = clean_value(
            row["Parent / Guardian Name"]
        )

        parent_mobile = clean_phone(
            row["Parent / Guardian Mobile Number"]
        )

        # --------------------------------------------------
        # Skip invalid rows
        # --------------------------------------------------

        if not register_number or not student_name:

            skipped += 1

            print(
                "SKIPPED: Missing register number/name"
            )

            continue

        # --------------------------------------------------
        # Check whether student already exists
        # --------------------------------------------------

        student = (
            db.query(Student)
            .filter(
                Student.register_number
                == register_number
            )
            .first()
        )

        # --------------------------------------------------
        # Update existing student
        # --------------------------------------------------

        if student:

            student.student_name = student_name
            student.department = department or "AI & DS"
            student.academic_batch = ACADEMIC_BATCH
            student.year = year
            student.section = section or "A"

            student.personal_email = personal_email
            student.phone_number = phone_number

            student.parent_guardian_name = parent_name
            student.parent_guardian_mobile = parent_mobile

            student.mentor_name = MENTOR_NAME
            student.mentor_phone = MENTOR_PHONE

            updated += 1

            print(
                f"UPDATED: {register_number} - {student_name}"
            )

            continue

        # --------------------------------------------------
        # Create new student
        # --------------------------------------------------

        new_student = Student(

            register_number=register_number,

            student_name=student_name,

            department=department or "AI & DS",

            academic_batch=ACADEMIC_BATCH,

            year=year,

            section=section or "A",

            # Excel may not contain college email,
            # so personal email is used when available.
            college_email=(
                personal_email
                or f"{register_number.lower()}@college.local"
            ),

            personal_email=personal_email,

            phone_number=phone_number,

            mentor_name=MENTOR_NAME,

            mentor_phone=MENTOR_PHONE,

            # These are required in the current model.
            # Excel import does not register face/video data.
            video_path="",

            profile_image_path=None,

            status="ACTIVE",

            passed_out_year=None,

            parent_guardian_name=parent_name,

            parent_guardian_mobile=parent_mobile,
        )

        db.add(new_student)

        inserted += 1

        print(
            f"INSERTED: {register_number} - {student_name}"
        )

    # ------------------------------------------------------
    # Commit
    # ------------------------------------------------------

    try:

        db.commit()

    except Exception:

        db.rollback()

        raise

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("IMPORT COMPLETED")
    print("=" * 70)

    print(f"Inserted : {inserted}")
    print(f"Updated  : {updated}")
    print(f"Skipped  : {skipped}")
    print(f"Total    : {inserted + updated}")

    print("=" * 70)

    return {
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "total": inserted + updated,
    }