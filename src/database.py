"""SQLite persistence for outreach history and duplicate protection."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/outreach.db")


def _connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create the local outreach log if it does not already exist."""
    with _connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS outreach_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                influencer_name TEXT NOT NULL,
                contact_email TEXT NOT NULL,
                email_subject TEXT,
                message_generated INTEGER NOT NULL DEFAULT 0,
                delivery_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                sent_at TEXT,
                error_message TEXT,
                UNIQUE(channel_id, contact_email)
            )
        """)


def already_contacted(channel_id: str, email: str, db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    """Return whether an address was already sent/simulated in any campaign row.

    The channel id is retained in the log for traceability, while checking by
    email prevents two discovered channels owned by the same creator receiving
    duplicate outreach in a single campaign run.
    """
    init_db(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM outreach_log WHERE contact_email = ? AND status IN ('sent', 'simulated')",
            (email.lower(),),
        ).fetchone()
    return row is not None


def log_outreach(record: dict[str, Any], db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Insert or update one event for outreach audit and campaign review."""
    init_db(db_path)
    values = (
        str(record.get("channel_id", "")).strip(), str(record.get("influencer_name", "")).strip(),
        str(record.get("contact_email", "")).strip().lower(), str(record.get("email_subject", "")).strip(),
        int(bool(record.get("message_generated"))), str(record.get("delivery_mode", "simulation")),
        str(record.get("status", "unknown")), record.get("sent_at"), str(record.get("error_message", "")).strip(),
    )
    with _connect(db_path) as connection:
        connection.execute("""
            INSERT INTO outreach_log (channel_id, influencer_name, contact_email, email_subject,
                message_generated, delivery_mode, status, sent_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(channel_id, contact_email) DO UPDATE SET
                influencer_name=excluded.influencer_name, email_subject=excluded.email_subject,
                message_generated=excluded.message_generated, delivery_mode=excluded.delivery_mode,
                status=excluded.status, sent_at=excluded.sent_at, error_message=excluded.error_message
        """, values)


def utc_now() -> str:
    """Return an ISO-8601 timestamp for the outreach tracker."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
