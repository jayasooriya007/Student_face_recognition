"""
test_bulk_embeddings.py

Generates individual 512-dimensional ArcFace embeddings
for every student whose aligned face images already exist.

IMPORTANT:
    SCRFD is NOT executed here.

Input:
    dataset/faces/<register_number>/

Output:
    dataset/embeddings/<register_number>.npy
    dataset/embeddings/<register_number>_images.npy

No averaging is performed.
Every aligned face gets its own embedding.
"""

from app.config.settings import FACE_DATASET_DIR

from app.face.embedding_service import EmbeddingService


# ==========================================================
# Initialize Embedding Service
# ==========================================================

embedding_service = EmbeddingService()


# ==========================================================
# Check Face Dataset Directory
# ==========================================================

if not FACE_DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Face dataset directory not found:\n"
        f"{FACE_DATASET_DIR}"
    )


# ==========================================================
# Find All Student Folders
# ==========================================================

student_folders = sorted(
    [
        folder
        for folder in FACE_DATASET_DIR.iterdir()
        if folder.is_dir()
    ]
)


# ==========================================================
# Header
# ==========================================================

print("=" * 70)

print(
    "IDENTIX - BULK ARCFACE EMBEDDING GENERATION"
)

print("=" * 70)

print()

print(
    "Input Directory:"
)

print(
    FACE_DATASET_DIR
)

print()

print(
    "Total Student Folders:",
    len(student_folders)
)

print()

print("=" * 70)


# ==========================================================
# Counters
# ==========================================================

successful_students = []

failed_students = []

total_faces = 0

total_embeddings = 0

total_failed_embeddings = 0


# ==========================================================
# Process Every Student
# ==========================================================

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

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        result = (
            embedding_service
            .generate_student_embeddings(
                register_number,
                student_folder,
            )
        )

        # --------------------------------------------------
        # Get result values
        # --------------------------------------------------

        faces_found = result[
            "faces_found"
        ]

        embeddings_generated = result[
            "embeddings_generated"
        ]

        embeddings_failed = result[
            "embeddings_failed"
        ]

        # --------------------------------------------------
        # Update totals
        # --------------------------------------------------

        total_faces += faces_found

        total_embeddings += (
            embeddings_generated
        )

        total_failed_embeddings += (
            embeddings_failed
        )

        # --------------------------------------------------
        # Store successful student
        # --------------------------------------------------

        successful_students.append(
            register_number
        )

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        print(
            "Faces Found:",
            faces_found
        )

        print(
            "Embeddings Generated:",
            embeddings_generated
        )

        print(
            "Embeddings Failed:",
            embeddings_failed
        )

        print(
            "Embedding Dimension:",
            result["embedding_dimension"]
        )

        print(
            "Embedding File:",
            result["embedding_file"]
        )

        print(
            "Status: SUCCESS"
        )

    except Exception as e:

        # --------------------------------------------------
        # Student failed
        # --------------------------------------------------

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
    "BULK ARCFACE EMBEDDING GENERATION COMPLETED"
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
    "Total Face Images:",
    total_faces
)

print(
    "Total Embeddings Generated:",
    total_embeddings
)

print(
    "Total Embeddings Failed:",
    total_failed_embeddings
)


# ==========================================================
# Successful Students
# ==========================================================

print()

print(
    "Successful Students:"
)

for register_number in successful_students:

    print(
        "  ✓",
        register_number
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
            "  ✗",
            register_number
        )

        print(
            "    Error:",
            error
        )


# ==========================================================
# Finish
# ==========================================================

print()

print("=" * 70)

print(
    "Embedding generation process finished."
)

print("=" * 70)