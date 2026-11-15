"""
import_students_from_excel.py

Imports student details from the Excel file into
the existing Identix MySQL students table.

Excel:
data/2023-2027-A.xlsx
Sheet:
Full details
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.database.models import Student


# ==========================================================
# Configuration
# ==========================================================

EXCEL_FILE = Path("data/2023-2027-A.xlsx")

SHEET_NAME = "Full details"

# The Excel sheet has its actual column headers on row 5.
# pandas uses zero-based indexing, therefore header=4.
HEADER_ROW = 4

MENTOR_NAME = "Stenni martin"
MENTOR_PHONE = "9500678584"

ACADEMIC_BATCH = "2023-2027"

# Students are currently in 4th year for the 2026-27
# academic year.
CURRENT_YEAR = 4


# ==========================================================
# Helper Functions
# ==========================================================

def clean_text(value):
    """
    Convert Excel values into clean strings.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return None

    return value


def clean_phone(value):
    """
    Convert Excel phone numbers such as:

        6369094326.0

    into:

        6369094326
    """

    if pd.isna(value):
        return None

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

    except (ValueError, TypeError):
        pass

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def clean_email(value):
    """
    Clean email values.
    """

    value = clean_text(value)

    if value:
        return value.lower()

    return None


# ==========================================================
# Read Excel
# ==========================================================

def load_excel():

    if not EXCEL_FILE.exists():

        raise FileNotFoundError(
            f"Excel file not found:\n{EXCEL_FILE}"
        )

    print("=" * 70)
    print("Reading Excel file")
    print("=" * 70)

    print(f"File : {EXCEL_FILE}")
    print(f"Sheet: {SHEET_NAME}")

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=SHEET_NAME,
        header=HEADER_ROW,
    )

    print(f"\nTotal Excel rows: {len(df)}")

    return df


# ==========================================================
# Import Students
# ==========================================================

def import_students():

    df = load_excel()

    db: Session = SessionLocal()

    inserted = 0
    skipped = 0
    failed = 0

    try:

        print("\n" + "=" * 70)
        print("IMPORTING STUDENTS")
        print("=" * 70)

        for index, row in df.iterrows():

            try:

                # --------------------------------------------------
                # Read Excel values
                # --------------------------------------------------

                register_number = clean_text(
                    row.get("Roll no")
                )

                student_name = clean_text(
                    row.get(
                        "Name of the student (in caps)"
                    )
                )

                department = clean_text(
                    row.get("Programme Code")
                )

                section = clean_text(
                    row.get("Section")
                )

                personal_email = clean_email(
                    row.get("Email id")
                )

                phone_number = clean_phone(
                    row.get(
                        "Student mobile number"
                    )
                )

                parent_name = clean_text(
                    row.get(
                        "Parent / Guardian Name"
                    )
                )

                parent_mobile = clean_phone(
                    row.get(
                        "Parent / Guardian Mobile Number"
                    )
                )

                # --------------------------------------------------
                # Validate required fields
                # --------------------------------------------------

                if not register_number:

                    print(
                        f"SKIPPED row {index + HEADER_ROW + 1}: "
                        "Missing Roll no"
                    )

                    skipped += 1
                    continue

                if not student_name:

                    print(
                        f"SKIPPED {register_number}: "
                        "Missing student name"
                    )

                    skipped += 1
                    continue

                # --------------------------------------------------
                # Check duplicate register number
                # --------------------------------------------------

                existing = (
                    db.query(Student)
                    .filter(
                        Student.register_number
                        == register_number
                    )
                    .first()
                )

                if existing:

                    print(
                        f"SKIPPED {register_number}: "
                        "Already exists"
                    )

                    skipped += 1
                    continue

                # --------------------------------------------------
                # Generate a safe placeholder video path
                #
                # These Excel records do not have registration
                # videos. Face data will be handled separately.
                # --------------------------------------------------

                video_path = (
                    f"dataset/registration_videos/"
                    f"{register_number}/"
                    f"{register_number}.mp4"
                )

                # --------------------------------------------------
                # Create Student object
                # --------------------------------------------------

                student = Student(

                    register_number=register_number,

                    student_name=student_name,

                    department=department
                    if department
                    else "AI & DS",

                    academic_batch=ACADEMIC_BATCH,

                    year=CURRENT_YEAR,

                    section=section
                    if section
                    else "A",

                    # We only have "Email id" in the Excel
                    # data we are importing.
                    #
                    # Keep college_email NULL for now.
                    # personal_email contains the Excel email.
                    college_email=None,

                    personal_email=personal_email,

                    phone_number=phone_number,

                    mentor_name=MENTOR_NAME,

                    mentor_phone=MENTOR_PHONE,

                    video_path=video_path,

                    profile_image_path=None,

                    status="ACTIVE",

                    passed_out_year=None,

                    parent_guardian_name=parent_name,

                    parent_guardian_mobile=parent_mobile,
                )

                db.add(student)

                inserted += 1

                print(
                    f"ADDED   {register_number:<18} "
                    f"{student_name}"
                )

            except Exception as e:

                failed += 1

                print(
                    f"FAILED row {index + HEADER_ROW + 1}: "
                    f"{e}"
                )

        # ------------------------------------------------------
        # Commit all successful records
        # ------------------------------------------------------

        db.commit()

        print("\n" + "=" * 70)
        print("IMPORT COMPLETED")
        print("=" * 70)

        print(f"Inserted : {inserted}")
        print(f"Skipped  : {skipped}")
        print(f"Failed   : {failed}")
        print(f"Excel    : {len(df)}")

        # ------------------------------------------------------
        # Verify database count
        # ------------------------------------------------------

        total_students = (
            db.query(Student).count()
        )

        print(
            f"\nTotal Students in MySQL: "
            f"{total_students}"
        )

        print("=" * 70)

    except Exception:

        db.rollback()

        raise

    finally:

        db.close()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    import_students()