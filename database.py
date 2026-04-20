"""SQLite storage for structured document data and comparison history."""

import sqlite3 
import json 
from datetime import datetime 
from typing import Optional 
from pathlib import Path 

DB_PATH = Path("rag_store.db")


def get_conn() -> sqlite3.Connection: 
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row #instead of returning rows as standard tuples, the connection will return sqlite3.Row objects
    # this helps accessing columns by their names rather than just their numerical index. 
    return conn


def init_db() -> None: 
    with get_conn() as conn: 
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY, 
            source_file TEXT NOT NULL, 
            extracted_text TEXT, 
            parsed_table_data TEXT, 
            parsed_chart_data TEXT, 
            parsed_equations TEXT, 
            created_at TEXT NOT NULL, 
            updated_at TEXT NOT NULL
        ); 

        CREATE TABLE IF NOT EXISTS comparison_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            document_id TEXT NOT NULL, 
            query TEXT NOT NULL, 
            response TEXT NOT NULL, 
            created_at TEXT NOT NULL, 
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        ); 
        """)


def upsert_document(
    document_id : str, 
    source_file : str, 
    extracted_text : str, 
    tables : list, 
    charts : list, 
    equations : list
) -> None: 
    now = datetime.utcnow().isoformat() 
    with get_conn() as conn: 
        conn.execute(
            """
            INSERT INTO documents
                (document_id, source_file, extracted_text, parsed_table_data, parsed_chart_data, parsed_equations, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                extracted_text = excluded.extracted_text, 
                parsed_chart_data = excluded.parsed_chart_data, 
                parsed_equations = excluded.parsed_equations, 
                parsed_table_data = excluded.parsed_table_data, 
                updated_at = excluded.updated_at
            """, 
            (
                document_id,
                source_file, 
                extracted_text, 
                json.dumps(tables), 
                json.dumps(charts), 
                json.dumps(equations), #sqlite cannot store python lists directly, json string works in that case 
                now, 
                now,
            ), 
        )


def get_document(document_id : str) -> Optional[sqlite3.Row]: 
    with get_conn() as conn: 
        return conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", [document_id]
        ).fetchone()


def log_comparison(document_id : str, query : str, response : str) -> None: 
    # what's the point of this method? 

    # if same query comes again, return the cached response instead of recomputing(optimization) 

    # debugging gets easier, i can track input query and output response 

    # chat history regarding a certain document 

    now = datetime.utcnow().isoformat()
    with get_conn() as conn: 
        conn.execute(
            "INSERT INTO comparison_history(document_id, query, response, created_at) VALUES (?, ?, ?, ?)", 
            (document_id, query, response, now), 
        )


def list_documents() -> list: 
    with get_conn() as conn: 
        return conn.execute(
            "SELECT document_id, source_file, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()