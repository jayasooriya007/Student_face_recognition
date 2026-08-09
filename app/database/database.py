"""
database.py

Creates the SQLAlchemy database engine,
session factory and ORM base.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config.settings import DATABASE_URL

# ======================================
# Database Engine
# ======================================

engine = create_engine(
    DATABASE_URL,
    echo=True,          # Shows SQL queries in terminal
    future=True,
    pool_pre_ping=True  # Automatically checks dead connections
)

# ======================================
# Session Factory
# ======================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ======================================
# ORM Base
# ======================================

Base = declarative_base()

# ======================================
# Dependency
# ======================================

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ======================================
# Create Tables
# ======================================

def create_tables():

    Base.metadata.create_all(bind=engine)