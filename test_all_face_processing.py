"""
test_all_face_processing.py

Processes all students that have quality-filtered
frames using SCRFD face detection.
"""

from app.face.face_processor import FaceProcessor


# ==========================================================
# Initialize Face Processor
# ==========================================================

processor = FaceProcessor()


# ==========================================================
# Get Student Folders
# ==========================================================

from app.config.settings import (
    QUALITY_FILTERED_DATASET_DIR,
)

student_folders = [
    folder
    for folder in QUALITY_FILTERED_DATASET_DIR.iterdir()
    if folder.is_dir()
]

student_folders.sort()


# ==========================================================
# Display Total Students
# ==========================================================

print("=" * 70)

print(
    "Total students with quality-filtered folders:",
    len(student_folders)
)

print("=" * 70)


# ==========================================================
# Statistics
# ==========================================================

successful_students = 0
students_without_frames = 0
failed_students = 0

total_images_checked = 0
total_faces_detected = 0
total_images_rejected = 0


# ==========================================================
# Process Every Student
# ==========================================================

for student_folder in student_folders:

    register_number = student_folder.name

    print()
    print("-" * 70)

    print(
        "Processing:",
        register_number
    )

    print("-" * 70)

    try:

        result = processor.process_student(
            register_number
        )

        total_images_checked += (
            result["total_images"]
        )

        total_faces_detected += (
            result["detected_faces"]
        )

        total_images_rejected += (
            result["rejected_images"]
        )

        if result["total_images"] == 0:

            students_without_frames += 1

            print(
                "❌ No quality-filtered images found."
            )

            continue

        successful_students += 1

        print(
            "Images checked:",
            result["total_images"]
        )

        print(
            "Faces detected:",
            result["detected_faces"]
        )

        print(
            "Images rejected:",
            result["rejected_images"]
        )

        print(
            "Faces saved:",
            result["output_folder"]
        )

    except Exception as error:

        failed_students += 1

        print(
            "❌ Error processing",
            register_number
        )

        print(
            "Error:",
            error
        )


# ==========================================================
# Final Summary
# ==========================================================

print()
print("=" * 70)

print("FACE PROCESSING COMPLETED")

print("=" * 70)

print(
    "Students processed:",
    successful_students
)

print(
    "Students without frames:",
    students_without_frames
)

print(
    "Students with errors:",
    failed_students
)

print(
    "Total images checked:",
    total_images_checked
)

print(
    "Total faces detected:",
    total_faces_detected
)

print(
    "Total images rejected:",
    total_images_rejected
)

print("=" * 70)
