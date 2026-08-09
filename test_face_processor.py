"""
test_face_processor.py

Tests SCRFD face detection and face alignment
for one student.
"""

from app.config.settings import (
    QUALITY_FILTERED_DATASET_DIR,
)

from app.face.face_processor import FaceProcessor


# ==========================================================
# Student To Test
# ==========================================================

REGISTER_NUMBER = "727723EUAI003"


# ==========================================================
# Initialize Face Processor
# ==========================================================

processor = FaceProcessor()


# ==========================================================
# Check Input Folder
# ==========================================================

input_folder = (
    QUALITY_FILTERED_DATASET_DIR
    / REGISTER_NUMBER
)


if not input_folder.exists():

    raise FileNotFoundError(
        f"Quality-filtered folder not found:\n"
        f"{input_folder}"
    )


# ==========================================================
# Display Information
# ==========================================================

print("=" * 60)

print(
    "Testing SCRFD face alignment for:",
    REGISTER_NUMBER
)

print(
    "Input folder:",
    input_folder
)

print("=" * 60)


# ==========================================================
# Process Student
# ==========================================================

result = processor.process_student(
    REGISTER_NUMBER
)


# ==========================================================
# Display Result
# ==========================================================

print()

print(
    "Register Number:",
    result["register_number"]
)

print(
    "Images Checked:",
    result["total_images"]
)

print(
    "Aligned Faces:",
    result["aligned_faces"]
)

print(
    "Images Rejected:",
    result["rejected_images"]
)

print(
    "Output Folder:",
    result["output_folder"]
)

print()

print("=" * 60)

print(
    "SCRFD face alignment completed."
)

print("=" * 60)