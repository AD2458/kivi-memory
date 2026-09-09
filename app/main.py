"""
FastAPI Application for the Kivi Phonetic Memory System.

Provides the HTTP API endpoints to ingest ASR observations, apply explicit
corrections, run the intervention pipeline, and inspect the internal
database/logs.
"""
import sqlite3
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Depends
from pydantic import BaseModel

from . import db
from . import learning
from . import intervention


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database on startup (create tables if missing)
    db.init_db()
    yield


app = FastAPI(
    title="Kivi Phonetic Memory System",
    description="Evidence-Gated Phonetic RAG Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


def get_db():
    """Dependency to provide a sqlite3 connection per request."""
    with db.db_session() as conn:
        yield conn


# --- Models ---
class ObserveRequest(BaseModel):
    asr_text: str
    formatted_text: str


class CorrectRequest(BaseModel):
    canonical_word: str
    context_snippet: str
    asr_token: Optional[str] = None


class OverrideRequest(BaseModel):
    asr_token: str
    canonical_word: str

class ProcessRequest(BaseModel):
    asr_text: str


# --- Endpoints ---

@app.post("/overrides", summary="Add an exact override rule")
def add_override(req: OverrideRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Saves a direct spelling override to bypass phonetic processing."""
    conn.execute(
        "INSERT OR REPLACE INTO exact_overrides (asr_token, canonical_word) VALUES (?, ?)",
        (req.asr_token, req.canonical_word)
    )
    return {"status": "success", "message": f"Added override {req.asr_token} -> {req.canonical_word}"}


@app.get("/overrides", summary="Get all exact overrides")
def get_overrides(conn: sqlite3.Connection = Depends(get_db)):
    """Returns the list of direct overrides."""
    rows = conn.execute("SELECT * FROM exact_overrides ORDER BY created_at DESC").fetchall()
    return {"overrides": [dict(r) for r in rows]}


@app.post("/learn/observe", summary="Ingest an implicit observation pair")
def learn_observe(req: ObserveRequest, conn: sqlite3.Connection = Depends(get_db)):
    """
    Diffs the raw ASR text against the corrected formatted text.
    Extracts token corrections and upserts them to the database.
    """
    results = learning.ingest_observation(conn, req.asr_text, req.formatted_text)
    return {"status": "success", "learned": results}


@app.post("/learn/correct", summary="Ingest an explicit user correction")
def learn_correct(req: CorrectRequest, conn: sqlite3.Connection = Depends(get_db)):
    """
    Directly asserts a correction without needing multiple observations.
    Bypasses the 'pending' state entirely.
    """
    result = learning.ingest_explicit_correction(
        conn, req.canonical_word, req.context_snippet, req.asr_token
    )
    return {"status": "success", "learned": result}


@app.post("/process", summary="Run the Intervention Pipeline")
async def process_asr(req: ProcessRequest, conn: sqlite3.Connection = Depends(get_db)):
    """
    Finds candidates, invokes the LLM Judge for semantic validation,
    applies string replacements if approved, and logs the outcome.
    """
    final_text = await intervention.process_transcript(conn, req.asr_text)
    return {"status": "success", "original": req.asr_text, "processed": final_text}


@app.get("/dictionary", summary="Inspect the active user dictionary")
def get_dictionary(conn: sqlite3.Connection = Depends(get_db)):
    """Returns the current state of the phonetic dictionary."""
    rows = conn.execute(
        "SELECT * FROM user_dictionary ORDER BY last_seen_at DESC"
    ).fetchall()
    return {"dictionary": [dict(r) for r in rows]}


@app.get("/logs", summary="View intervention trace logs")
def get_logs(limit: int = 50, conn: sqlite3.Connection = Depends(get_db)):
    """Returns the most recent LLM judgments and string replacements."""
    rows = conn.execute(
        "SELECT * FROM intervention_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return {"logs": [dict(r) for r in rows]}


@app.delete("/dictionary/all", summary="Clear entire memory store")
def clear_all_memory(conn: sqlite3.Connection = Depends(get_db)):
    """Deletes all data from the database."""
    conn.execute("DELETE FROM memory_evidence")
    conn.execute("DELETE FROM user_dictionary")
    conn.execute("DELETE FROM intervention_logs")
    return {"status": "success", "message": "All data cleared."}


@app.delete("/dictionary/{memory_id}", summary="Delete specific memory")
def delete_memory(memory_id: int, conn: sqlite3.Connection = Depends(get_db)):
    """Deletes a specific entry by its ID."""
    conn.execute("DELETE FROM memory_evidence WHERE dictionary_id = ?", (memory_id,))
    conn.execute("DELETE FROM user_dictionary WHERE id = ?", (memory_id,))
    return {"status": "success", "message": f"Deleted memory {memory_id}."}


