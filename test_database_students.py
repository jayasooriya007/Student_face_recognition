"""
test_database_students.py

Checks whether the students used by FAISS
actually exist in the MySQL students table.
"""

from app.database.database import SessionLocal
from app.database.models import Student


db = SessionLocal()

try:

    students = (
        db.query(Student)
        .order_by(Student.register_number)
        .all()
    )

    print("=" * 70)
    print("DATABASE STUDENT CHECK")
    print("=" * 70)

    print()

    print(
        "Total Students in MySQL:",
        len(students)
    )

    print()

    for student in students:

        print(
            student.register_number,
            "|",
            student.name,
            "|",
            student.department
        )

    print()

    print("=" * 70)

finally:

    db.close()