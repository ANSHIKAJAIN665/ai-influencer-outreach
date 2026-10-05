"""Safe email delivery/simulation with SQLite-based duplicate protection."""

import logging
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:  # Supports both `python app.py` and `python src/sender.py`.
    from src.database import already_contacted, init_db, log_outreach, utc_now
except ModuleNotFoundError:
    from database import already_contacted, init_db, log_outreach, utc_now

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_CSV = Path("data/influencers_personalized.csv")
TRACKER_CSV = Path("data/outreach_tracker.csv")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
SEND_DELAY = float(os.getenv("SEND_DELAY", "3"))


def _valid_email(value: object) -> bool:
    email = str(value or "").strip().lower()
    return "@" in email and email not in {"not found", "nan", "none"}


def _deliver(email: str, subject: str, message: str) -> tuple[bool, str]:
    if DRY_RUN:
        log.info("[SIMULATED] %s", email)
        return True, "simulated"
    try:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = SENDER_EMAIL, email, subject
        msg.set_content(message)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True, "sent"
    except Exception as exc:
        return False, str(exc)


def run_sender(input_path: Path = INPUT_CSV) -> dict[str, int]:
    """Send or safely simulate valid, newly-generated outreach messages."""
    if not input_path.exists():
        raise FileNotFoundError(f"Personalized dataset not found: {input_path}")
    if not DRY_RUN and (not SENDER_EMAIL or not SENDER_PASSWORD):
        raise ValueError("SENDER_EMAIL and SENDER_PASSWORD are required when DRY_RUN=false")

    df = pd.read_csv(input_path).fillna("")
    required = {"name", "channel_id", "contact_email", "email_subject", "email_pitch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Personalized CSV missing columns: {sorted(missing)}")
    init_db()
    tracker = []
    result = {"total": 0, "successful": 0, "failed": 0, "skipped": 0}

    for _, row in df.iterrows():
        email, channel_id = str(row.contact_email).strip(), str(row.channel_id).strip()
        # `row.name` is Pandas' numeric row index; the creator name must come
        # from the dataset field so both the tracker and SQLite audit are correct.
        name = str(row["name"]).strip()
        subject, message = str(row.email_subject).strip(), str(row.email_pitch).strip()
        if not _valid_email(email):
            result["skipped"] += 1
            continue
        result["total"] += 1
        base = {"channel_id": channel_id, "influencer_name": name, "contact_email": email,
                "email_subject": subject, "message_generated": bool(message),
                "delivery_mode": "simulation" if DRY_RUN else "smtp"}
        if already_contacted(channel_id, email):
            base.update(status="skipped_duplicate", sent_at=None, error_message="Already logged as sent/simulated")
            # Do not overwrite the successful delivery record: it is the
            # evidence used to block duplicates on later runs.
            tracker.append(base)
            result["skipped"] += 1
            continue
        if not message:
            base.update(status="failed", sent_at=None, error_message="No generated email message")
            log_outreach(base)
            tracker.append(base)
            result["failed"] += 1
            continue
        ok, detail = _deliver(email, subject, message)
        base.update(status=detail if ok else "failed", sent_at=utc_now() if ok else None,
                    error_message="" if ok else detail)
        log_outreach(base)
        tracker.append(base)
        result["successful" if ok else "failed"] += 1
        if ok and not DRY_RUN:
            time.sleep(SEND_DELAY)

    columns = ["influencer_name", "contact_email", "channel_id", "email_subject", "message_generated", "delivery_mode", "sent_at", "status", "error_message"]
    pd.DataFrame(tracker, columns=columns).to_csv(TRACKER_CSV, index=False)
    log.info("Outreach complete: %s", result)
    return result


if __name__ == "__main__":
    print(run_sender())
