"""
Initialize database with pgvector extension.
Run this script once to set up the database.
"""
from app.database import init_db, engine
from sqlalchemy import text

if __name__ == "__main__":
    print("Initializing database...")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    init_db()
    print("Database initialized successfully!")

