from __future__ import annotations
"""Local SQLite database for cogcore-specific data."""

import aiosqlite
import config

_db_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """Get or create the database connection."""
    global _db_connection
    if _db_connection is None:
        _db_connection = await aiosqlite.connect(config.DB_PATH)
        _db_connection.row_factory = aiosqlite.Row
        await _init_tables(_db_connection)
    return _db_connection


async def close_db():
    """Close the database connection."""
    global _db_connection
    if _db_connection is not None:
        await _db_connection.close()
        _db_connection = None


async def _init_tables(db: aiosqlite.Connection):
    """Create all tables if they don't exist."""
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_embedding BLOB NOT NULL,
            action TEXT NOT NULL,
            action_details TEXT,
            outcome_embedding BLOB NOT NULL,
            was_successful INTEGER NOT NULL,
            prediction_error REAL,
            prediction_id TEXT UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS principles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            principle TEXT NOT NULL,
            evidence_count INTEGER DEFAULT 0,
            domains TEXT,
            confidence REAL DEFAULT 0.5,
            actionable_rule TEXT,
            times_applied INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS thought_nodes (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            thought TEXT NOT NULL,
            thought_type TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            parent_id TEXT,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME
        );

        CREATE TABLE IF NOT EXISTS thought_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relationship TEXT NOT NULL,
            strength REAL DEFAULT 0.5,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS curiosity_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL UNIQUE,
            prediction_error_avg REAL DEFAULT 0,
            rnd_novelty_avg REAL DEFAULT 0,
            info_gain_avg REAL DEFAULT 0,
            combined_score REAL DEFAULT 0,
            sample_count INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS self_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_json TEXT NOT NULL,
            capability_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS model_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            fisher_diagonal BLOB,
            anchor_params BLOB,
            stats_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    await db.commit()
