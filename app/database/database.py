"""
database.py

Creates the SQLAlchemy database engine, session,
and base class for ORM models.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config.settings import DATABASE_URL


# ==============================
# Database Engine
# ==============================

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True
)


# ==============================
# Session Factory
# ==============================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ==============================
# Base Class
# ==============================

Base = declarative_base()


# ==============================
# Dependency for FastAPI
# ==============================

def get_db():
    """
    Creates a database session.

    Automatically closes the session
    after every request.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ==============================
# Create Tables
# ==============================

def create_tables():
    """
    Creates all database tables.
    """

    Base.metadata.create_all(bind=engine)