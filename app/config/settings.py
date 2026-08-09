"""
settings.py

Central configuration file for the Identix project.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


# ==========================================================
# Base Project Directory
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv(BASE_DIR / ".env")


# ==========================================================
# Original Student Dataset
# ==========================================================

STUDENT_DATASET_DIR = Path(
    r"C:\Users\suriy\Desktop\Student_dataset"
)


# ==========================================================
# Dataset Directories
# ==========================================================

DATASET_DIR = BASE_DIR / "dataset"

RAW_DATASET_DIR = DATASET_DIR / "raw"

PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

QUALITY_FILTERED_DATASET_DIR = (
    DATASET_DIR / "quality_filtered"
)

FACE_DATASET_DIR = DATASET_DIR / "faces"

EMBEDDINGS_DIR = DATASET_DIR / "embeddings"


# ==========================================================
# Upload Directories
# ==========================================================

UPLOAD_DIR = BASE_DIR / "uploads"

VIDEO_UPLOAD_DIR = UPLOAD_DIR / "videos"


# ==========================================================
# AI Model Directories
# ==========================================================

MODELS_DIR = BASE_DIR / "models"

SCRFD_MODEL_DIR = MODELS_DIR / "scrfd"

ARCFACE_MODEL_DIR = MODELS_DIR / "arcface"

FAISS_MODEL_DIR = MODELS_DIR / "faiss"


# ==========================================================
# Face Recognition Model Settings
# ==========================================================

MODEL_NAME = "buffalo_l"

DETECTION_SIZE = (640, 640)

USE_GPU = False

EMBEDDING_SIZE = 512

SIMILARITY_THRESHOLD = 0.40


# ==========================================================
# Database Configuration
# ==========================================================

DB_USER = os.getenv("DB_USER")

DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_HOST = os.getenv("DB_HOST")

DB_PORT = os.getenv("DB_PORT")

DB_NAME = os.getenv("DB_NAME")


DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ==========================================================
# Supported File Types
# ==========================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
]

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
]


# ==========================================================
# Auto Create Directories
# ==========================================================


DIRECTORIES = [
    DATASET_DIR,
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    QUALITY_FILTERED_DATASET_DIR,
    FACE_DATASET_DIR,
    EMBEDDINGS_DIR,
    UPLOAD_DIR,
    VIDEO_UPLOAD_DIR,
    MODELS_DIR,
    SCRFD_MODEL_DIR,
    ARCFACE_MODEL_DIR,
    FAISS_MODEL_DIR,
]


for directory in DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )