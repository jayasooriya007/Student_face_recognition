"""
test_build_faiss.py

Builds a fresh FAISS index from all student
ArcFace embeddings.

Input:
    dataset/embeddings/<register_number>.npy

Output:
    models/faiss/student.index
    models/faiss/student_ids.npy

Each individual embedding is stored separately.

Example:

    Student A -> 127 embeddings
    Student B -> 150 embeddings

All 277 embeddings are stored individually
inside the same FAISS index.
"""

from pathlib import Path

import numpy as np

from app.config.settings import EMBEDDINGS_DIR

from app.face.faiss_manager import FAISSManager


# ==========================================================
# Initialize
# ==========================================================

faiss_manager = FAISSManager()


# ==========================================================
# IMPORTANT:
# Start with a completely fresh FAISS index.
# ==========================================================

faiss_manager.clear()


# ==========================================================
# Check Embeddings Directory
# ==========================================================

if not EMBEDDINGS_DIR.exists():

    raise FileNotFoundError(
        f"Embeddings directory not found:\n"
        f"{EMBEDDINGS_DIR}"
    )


# ==========================================================
# Find Student Embedding Files
# ==========================================================

embedding_files = sorted(
    [
        file
        for file in EMBEDDINGS_DIR.glob("*.npy")
        if not file.name.endswith("_images.npy")
    ]
)


print("=" * 70)

print(
    "IDENTIX - BUILDING FAISS INDEX"
)

print("=" * 70)

print()

print(
    "Embeddings Directory:"
)

print(
    EMBEDDINGS_DIR
)

print()

print(
    "Student Embedding Files:",
    len(embedding_files)
)

print()

print("=" * 70)


# ==========================================================
# Counters
# ==========================================================

successful_students = []

failed_students = []

total_embeddings_added = 0


# ==========================================================
# Process Every Student
# ==========================================================

for index, embedding_file in enumerate(
    embedding_files,
    start=1,
):

    # ------------------------------------------------------
    # Register number comes from filename
    #
    # Example:
    #
    # 727723euai003.npy
    #
    # -> 727723euai003
    # ------------------------------------------------------

    register_number = (
        embedding_file.stem
    )

    print()

    print(
        f"[{index}/{len(embedding_files)}] "
        f"Adding: {register_number}"
    )

    print(
        "-" * 70
    )

    try:

        # --------------------------------------------------
        # Load embeddings
        # --------------------------------------------------

        embeddings = np.load(
            embedding_file
        )

        print(
            "Embedding Shape:",
            embeddings.shape
        )

        # --------------------------------------------------
        # Add to FAISS
        # --------------------------------------------------

        added = (
            faiss_manager
            .add_embeddings(
                register_number,
                embeddings,
                save=False,
            )
        )

        total_embeddings_added += added

        successful_students.append(
            register_number
        )

        print(
            "Embeddings Added:",
            added
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
# Save Once After All Students
# ==========================================================

faiss_manager.save()


# ==========================================================
# Final Verification
# ==========================================================

print()

print("=" * 70)

print(
    "FAISS INDEX BUILD COMPLETED"
)

print("=" * 70)

print()

print(
    "Student Files Found:",
    len(embedding_files)
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
    "Total Embeddings Added:",
    total_embeddings_added
)

print(
    "FAISS Embedding Count:",
    faiss_manager.get_embedding_count()
)

print(
    "FAISS Student Count:",
    faiss_manager.get_student_count()
)

print()

print(
    "FAISS Index:"
)

print(
    faiss_manager.index_path
)

print()

print(
    "Student Mapping:"
)

print(
    faiss_manager.id_path
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


# ==========================================================
# Final Status
# ==========================================================

print()

print("=" * 70)

if (
    len(failed_students) == 0
    and total_embeddings_added
    == faiss_manager.get_embedding_count()
):

    print(
        "FAISS BUILD: SUCCESS"
    )

else:

    print(
        "FAISS BUILD: CHECK ERRORS ABOVE"
    )

print("=" * 70)