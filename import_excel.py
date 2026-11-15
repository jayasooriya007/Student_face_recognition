from app.database.database import SessionLocal
from app.services.excel_import_service import import_students_from_excel


def main():

    db = SessionLocal()

    try:

        result = import_students_from_excel(db)

        print()
        print("Final result:")
        print(result)

    finally:

        db.close()


if __name__ == "__main__":
    main()