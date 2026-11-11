"""
SQLite Database Layer.
Manages persistent storage for job postings and resume scanning history.
"""

from typing import List, Dict, Any, Optional
import sqlite3
import os
import json
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "jobs.db")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Establish connection to SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DB_PATH) -> None:
    """Initialize database tables for jobs and scan history."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Table 1: Job listings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                category TEXT NOT NULL,
                location TEXT NOT NULL,
                experience_level TEXT NOT NULL,
                required_skills TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: Resume scan audit history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resume_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                skills_count INTEGER NOT NULL,
                skills_list TEXT NOT NULL,
                top_job_match TEXT,
                match_score REAL,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def populate_jobs_from_dataframe(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Populate jobs table from a pandas DataFrame, replacing existing listings."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs")
        
        for _, row in df.iterrows():
            skills_str = row["required_skills"]
            if isinstance(skills_str, list):
                skills_str = ", ".join(skills_str)

            cursor.execute("""
                INSERT INTO jobs (title, company, category, location, experience_level, required_skills, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row["title"],
                row["company"],
                row["category"],
                row.get("location", "Remote"),
                row.get("experience_level", "Mid-Level"),
                skills_str,
                row["description"]
            ))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        count = cursor.fetchone()[0]
        return count


def fetch_all_jobs(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all jobs from the SQLite database."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY id ASC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def log_resume_scan(
    filename: str,
    predicted_category: str,
    confidence: float,
    skills: List[str],
    top_job_match: Optional[str] = None,
    match_score: Optional[float] = None,
    db_path: str = DB_PATH
) -> int:
    """Log an analysis scan into the database."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO resume_scans (
                filename, predicted_category, confidence, skills_count, skills_list, top_job_match, match_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            filename,
            predicted_category,
            round(confidence, 4),
            len(skills),
            json.dumps(skills),
            top_job_match,
            round(match_score, 4) if match_score is not None else None
        ))
        conn.commit()
        return cursor.lastrowid


def fetch_scan_history(limit: int = 10, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Fetch recent resume scans from database."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filename, predicted_category, confidence, skills_count, top_job_match, match_score, scanned_at
            FROM resume_scans
            ORDER BY scanned_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

