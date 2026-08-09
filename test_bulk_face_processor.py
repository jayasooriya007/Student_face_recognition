"""
test_bulk_face_processor.py

Runs SCRFD face detection and 112x112 face alignment
for every student that has a quality-filtered folder.

This stage DOES NOT generate ArcFace embeddings.

It creates aligned face images in:

dataset/faces/<register_number>/
"""

from app.config.settings import (
    QUALITY_FILTERED_DATASET_DIR,
)

from app.face.face_processor import FaceProcessor


# ==========================================================
# Initialize
# ==========================================================

processor = FaceProcessor()


# ==========================================================
# Check Dataset
# ==========================================================

if not QUALITY_FILTERED_DATASET_DIR.exists():

    raise FileNotFoundError(
        "Quality-filtered dataset directory not found:\n"
        f"{QUALITY_FILTERED_DATASET_DIR}"
    )


# ==========================================================
# Find Student Folders
# ==========================================================

student_folders = sorted(
    [
        folder
        for folder in QUALITY_FILTERED_DATASET_DIR.iterdir()
        if folder.is_dir()
    ]
)


print("=" * 70)

print(
    "IDENTIX - BULK FACE ALIGNMENT"
)

print("=" * 70)

print()

print(
    "Input Directory:"
)

print(
    QUALITY_FILTERED_DATASET_DIR
)

print()

print(
    "Total Student Folders:",
    len(student_folders)
)

print()

print("=" * 70)


# ==========================================================
# Process Students
# ==========================================================

successful_students = []

failed_students = []

total_images = 0

total_aligned_faces = 0

total_rejected = 0


for index, student_folder in enumerate(
    student_folders,
    start=1,
):

    register_number = student_folder.name

    print()

    print(
        f"[{index}/{len(student_folders)}] "
        f"Processing: {register_number}"
    )

    print(
        "-" * 70
    )

    try:

        result = processor.process_student(
            register_number
        )

        images_checked = result[
            "total_images"
        ]

        aligned_faces = result[
            "aligned_faces"
        ]

        rejected = result[
            "rejected_images"
        ]

        total_images += images_checked

        total_aligned_faces += aligned_faces

        total_rejected += rejected

        successful_students.append(
            register_number
        )

        print(
            "Images Checked:",
            images_checked
        )

        print(
            "Aligned Faces:",
            aligned_faces
        )

        print(
            "Images Rejected:",
            rejected
        )

        print(
            "Output Folder:",
            result["output_folder"]
        )

        print(
            "Status: SUCCESS"
        )

    except Exception as e:

        failed_students.append(
            (
                register_number,
                str(e),
            )
        )

        print(
            "Status: FAILED"
        )

        print(
            "Error:",
            str(e)
        )


# ==========================================================
# Final Summary
# ==========================================================

print()

print("=" * 70)

print(
    "BULK FACE ALIGNMENT COMPLETED"
)

print("=" * 70)

print()

print(
    "Total Student Folders:",
    len(student_folders)
)

print(
    "Successful Students:",
    len(successful_students)
)

print(
    "Failed Students:",
    len(failed_students)
)

print(
    "Total Images Checked:",
    total_images
)

print(
    "Total Aligned Faces:",
    total_aligned_faces
)

print(
    "Total Rejected Images:",
    total_rejected
)


# ==========================================================
# Failed Students
# ==========================================================

if failed_students:

    print()

    print(
        "Failed Students:"
    )

    for register_number, error in (
        failed_students
    ):

        print()

        print(
            "Register Number:",
            register_number
        )

        print(
            "Error:",
            error
        )


print()

print("=" * 70)

print(
    "All available face alignment processing is complete."
)

print("=" * 70)