"""
test_import_excel.py

Imports student information from the Excel file
into the Identix MySQL students table.

Excel:
data/2023-2027-A.xlsx
Sheet:
Full details
"""

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.database.models import Student


# ==========================================================
# Configuration
# ==========================================================

EXCEL_FILE = "data/2023-2027-A.xlsx"
SHEET_NAME = "Full details"

MENTOR_NAME = "Stenni martin"
MENTOR_PHONE = "9500678584"

# Students without a registered face/video yet
PLACEHOLDER_VIDEO_PATH = "EXCEL_IMPORT"


# ==========================================================
# Helper Functions
# ==========================================================

def clean_value(value):
    """
    Convert Excel values into clean Python values.
    """

    if pd.isna(value):
        return None

    return str(value).strip()


def clean_phone(value):
    """
    Clean phone numbers coming from Excel.

    Excel may read numbers such as:
    6369094326.0

    and convert them to:
    6369094326
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def calculate_year(batch):
    """
    Calculate current academic year from batch.

    Example:

    2023-2027
    in 2026
    -> 4th year

    The project currently uses 2026 as the academic year.
    """

    try:
        start_year = int(str(batch).split("-")[0])
        current_year = datetime.now().year

        year = current_year - start_year + 1

        # Keep within our database validation range.
        if year < 1:
            year = 1

        if year > 4:
            year = 4

        return year

    except Exception:
        return 1


# ==========================================================
# Main Import Function
# ==========================================================

def import_students():

    print("=" * 70)
    print("IDENTIX - EXCEL STUDENT IMPORT")
    print("=" * 70)

    # ------------------------------------------------------
    # Read Excel
    # ------------------------------------------------------

    print("\nReading Excel file...")

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=SHEET_NAME,
        header=4,
    )

    print(f"Excel students found: {len(df)}")

    # ------------------------------------------------------
    # Validate required columns
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Required Excel columns are missing:")

        for column in missing_columns:
            print(f"  - {column}")

        return

    # ------------------------------------------------------
    # Database
    # ------------------------------------------------------

    db: Session = SessionLocal()

    inserted = 0
    updated = 0
    skipped = 0
    failed = 0

    try:

        # ==================================================
        # Process Every Student
        # ==================================================

        for index, row in df.iterrows():

            excel_row = index + 6

            try:

                # ------------------------------------------
                # Read Excel values
                # ------------------------------------------

                register_number = clean_value(
                    row["Roll no"]
                )

                student_name = clean_value(
                    row["Name of the student (in caps)"]
                )

                degree = clean_value(
                    row["Degree"]
                )

                programme = clean_value(
                    row["Programme Code"]
                )

                section = clean_value(
                    row["Section"]
                )

                college_email = clean_value(
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

                # ------------------------------------------
                # Basic validation
                # ------------------------------------------

                if not register_number:

                    print(
                        f"[SKIP] Excel row {excel_row}: "
                        "Missing register number"
                    )

                    skipped += 1
                    continue

                if not student_name:

                    print(
                        f"[SKIP] {register_number}: "
                        "Missing student name"
                    )

                    skipped += 1
                    continue

                # ------------------------------------------
                # Academic information
                # ------------------------------------------

                academic_batch = "2023-2027"

                year = calculate_year(
                    academic_batch
                )

                # ------------------------------------------
                # Department
                #
                # The Excel contains:
                #
                # Degree = B.Tech
                # Programme Code = AI & DS
                #
                # We store the programme as department.
                # ------------------------------------------

                department = programme or degree or "AI & DS"

                # ------------------------------------------
                # Find existing student
                # ------------------------------------------

                student = (
                    db.query(Student)
                    .filter(
                        Student.register_number
                        == register_number
                    )
                    .first()
                )

                # ==================================================
                # Existing Student
                # ==================================================

                if student:

                    student.student_name = student_name
                    student.department = department
                    student.academic_batch = academic_batch
                    student.year = year
                    student.section = section

                    # Only update email when available.
                    if college_email:
                        student.college_email = (
                            college_email
                        )

                    student.phone_number = phone_number

                    student.parent_guardian_name = (
                        parent_name
                    )

                    student.parent_guardian_mobile = (
                        parent_mobile
                    )

                    student.mentor_name = MENTOR_NAME
                    student.mentor_phone = MENTOR_PHONE

                    # Do NOT overwrite an existing real
                    # video path.
                    if not student.video_path:
                        student.video_path = (
                            PLACEHOLDER_VIDEO_PATH
                        )

                    student.updated_at = datetime.utcnow()

                    updated += 1

                    print(
                        f"[UPDATED] "
                        f"{register_number} - "
                        f"{student_name}"
                    )

                # ==================================================
                # New Student
                # ==================================================

                else:

                    new_student = Student(

                        register_number=(
                            register_number
                        ),

                        student_name=(
                            student_name
                        ),

                        department=(
                            department
                        ),

                        academic_batch=(
                            academic_batch
                        ),

                        year=year,

                        section=(
                            section
                        ),

                        college_email=(
                            college_email
                        ),

                        personal_email=None,

                        phone_number=(
                            phone_number
                        ),

                        mentor_name=(
                            MENTOR_NAME
                        ),

                        mentor_phone=(
                            MENTOR_PHONE
                        ),

                        video_path=(
                            PLACEHOLDER_VIDEO_PATH
                        ),

                        profile_image_path=None,

                        status="ACTIVE",

                        passed_out_year=None,

                    )

                    db.add(new_student)

                    inserted += 1

                    print(
                        f"[INSERTED] "
                        f"{register_number} - "
                        f"{student_name}"
                    )

            except Exception as e:

                failed += 1

                print(
                    f"[FAILED] Excel row "
                    f"{excel_row}: {e}"
                )

        # --------------------------------------------------
        # Commit everything
        # --------------------------------------------------

        db.commit()

        # ==================================================
        # Summary
        # ==================================================

        print("\n" + "=" * 70)
        print("IMPORT COMPLETED")
        print("=" * 70)

        print(
            f"Excel students      : {len(df)}"
        )

        print(
            f"New students        : {inserted}"
        )

        print(
            f"Existing updated    : {updated}"
        )

        print(
            f"Skipped             : {skipped}"
        )

        print(
            f"Failed              : {failed}"
        )

        total = (
            db.query(Student).count()
        )

        print(
            f"\nTotal students in MySQL: {total}"
        )

        print("=" * 70)

    except Exception as e:

        db.rollback()

        print("\nIMPORT FAILED")
        print(f"Error: {e}")

    finally:

        db.close()


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    import_students()