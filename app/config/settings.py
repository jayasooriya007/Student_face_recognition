from pathlib import Path


# ==============================
# Base Project Directory
# ==============================

BASE_DIR = Path(__file__).resolve().parents[2]


# ==============================
# Dataset Directories
# ==============================

DATASET_DIR = BASE_DIR / "dataset"

RAW_DATASET_DIR = DATASET_DIR / "raw"

PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

EMBEDDINGS_DIR = DATASET_DIR / "embeddings"


# ==============================
# Upload Directories
# ==============================

UPLOAD_DIR = BASE_DIR / "uploads"

VIDEO_UPLOAD_DIR = UPLOAD_DIR / "videos"


# ==============================
# AI Model Directories
# ==============================

MODELS_DIR = BASE_DIR / "models"

SCRFD_MODEL_DIR = MODELS_DIR / "scrfd"

ARCFACE_MODEL_DIR = MODELS_DIR / "arcface"

FAISS_MODEL_DIR = MODELS_DIR / "faiss"


# ==============================
# Database
# ==============================

DATABASE_URL = "postgresql://postgres:password@localhost:5432/student_db"


# ==============================
# Supported File Types
# ==============================

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]


# ==============================
# Face Recognition
# ==============================

EMBEDDING_SIZE = 512

SIMILARITY_THRESHOLD = 0.50


# ==============================
# Auto Create Directories
# ==============================

DIRECTORIES = [
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    EMBEDDINGS_DIR,
    VIDEO_UPLOAD_DIR,
    SCRFD_MODEL_DIR,
    ARCFACE_MODEL_DIR,
    FAISS_MODEL_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)