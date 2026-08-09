from app.config.settings import FACE_DATASET_DIR

from app.face.embedding_service import (
    EmbeddingService,
)


REGISTER_NUMBER = "727723EUAI003"


faces_folder = (
    FACE_DATASET_DIR
    / REGISTER_NUMBER
)


if not faces_folder.exists():

    raise FileNotFoundError(
        f"Face folder not found:\n"
        f"{faces_folder}"
    )


service = EmbeddingService()


print("=" * 60)

print(
    "Generating ArcFace embeddings for:",
    REGISTER_NUMBER
)

print(
    "Face folder:",
    faces_folder
)

print("=" * 60)


result = (
    service.generate_student_embeddings(
        REGISTER_NUMBER,
        faces_folder,
    )
)


print()

print(
    "Register Number:",
    result["register_number"]
)

print(
    "Faces Found:",
    result["faces_found"]
)

print(
    "Embeddings Generated:",
    result["embeddings_generated"]
)

print(
    "Embeddings Failed:",
    result["embeddings_failed"]
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
    "Image Names File:",
    result["image_names_file"]
)

print()

print("=" * 60)

print(
    "ArcFace embedding generation completed."
)

print("=" * 60)