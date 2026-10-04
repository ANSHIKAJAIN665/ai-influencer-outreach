# src/personalization.py

"""
Personalization Module
======================
Generates personalized outreach messages for qualified influencers.

Uses Gemini when available.
Automatically falls back to local personalization if Gemini fails.
"""

import os
import logging
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


# =========================================================
# Configuration
# =========================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = BASE_DIR / "data" / "influencers_enriched.csv"
OUTPUT_CSV = BASE_DIR / "data" / "influencers_personalized.csv"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Gemini is optional.
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"


# =========================================================
# Logging
# =========================================================

log = logging.getLogger(__name__)


# =========================================================
# Gemini Client
# =========================================================

client = None

if USE_GEMINI and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        log.info("Gemini client initialized.")
    except Exception as e:
        log.warning("Could not initialize Gemini: %s", e)
        client = None
else:
    log.info("Gemini disabled or API key not found. Using fallback.")


# =========================================================
# Helper Functions
# =========================================================

def clean_text(value, default=""):
    """Safely convert a value to clean text."""

    if value is None:
        return default

    if pd.isna(value):
        return default

    return str(value).strip()


def format_subscribers(value):
    """Format subscriber count safely."""

    try:
        return f"{int(float(value)):,}"
    except (ValueError, TypeError):
        return "your"


def get_themes(influencer):
    """
    Extract content themes from enriched influencer data.
    """

    themes = clean_text(
        influencer.get("content_themes", ""),
        "",
    )

    if not themes:
        return "AI, machine learning, and technology"

    # Convert comma-separated themes into readable text
    parts = [
        item.strip()
        for item in themes.split(",")
        if item.strip()
    ]

    if not parts:
        return "AI, machine learning, and technology"

    # Three signals are enough to feel specific while keeping the outreach concise.
    parts = parts[:3]

    if len(parts) == 1:
        return parts[0]

    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"

    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


# =========================================================
# Fallback Personalization
# =========================================================

def generate_fallback_message(influencer):
    """
    Generate a personalized message without using Gemini.

    This guarantees the pipeline can continue even when:
    - Gemini quota is exceeded
    - Gemini API is unavailable
    - network request fails
    """

    name = clean_text(
        influencer.get("name"),
        "there",
    )

    niche = clean_text(
        influencer.get("niche"),
        "AI and technology",
    )

    subscribers = format_subscribers(
        influencer.get("subscribers", 0)
    )

    themes = get_themes(influencer)

    subject = (
        f"Collaboration opportunity for "
        f"{name} – {niche}"
    )

    email_pitch = f"""Hi {name},

Your {niche} videos, especially the work around {themes}, align well with an upcoming creator campaign we are planning. With a community of about {subscribers} subscribers, you could help turn useful ideas into practical, audience-first content. We would love to discuss a paid sponsored tutorial or UGC-style collaboration, with creative freedom and clear campaign support. Would you be open to a short conversation about the fit and deliverables?

Best,
AI Influencer Outreach Team"""

    instagram_dm = (
        f"Hi {name}! Your {niche} content on {themes} is a great fit for our "
        f"paid creator campaign. Open to details?"
    )

    return {
        "email_subject": subject,
        "email_pitch": email_pitch,
        "instagram_dm": instagram_dm,
        "personalization_source": "fallback",
    }


# =========================================================
# Gemini Personalization
# =========================================================

def generate_gemini_message(influencer):
    """
    Generate personalized outreach using Gemini.

    Returns None if Gemini fails.
    """

    if client is None:
        return None

    name = clean_text(
        influencer.get("name"),
        "there",
    )

    niche = clean_text(
        influencer.get("niche"),
        "AI and technology",
    )

    subscribers = format_subscribers(
        influencer.get("subscribers", 0)
    )

    themes = get_themes(influencer)

    prompt = f"""
Create a professional but friendly influencer outreach message.

Influencer:
Name: {name}
Niche: {niche}
Subscribers: {subscribers}
Content themes: {themes}

Return exactly this format:

SUBJECT: <short email subject>

EMAIL: <personalized email body>

DM: <short Instagram/YouTube DM>

Rules:
- Do not invent facts about the influencer.
- Mention their actual niche and content themes.
- Email body must be 60–90 words, excluding greeting and sign-off.
- DM must be 15–30 words.
- Do not use exaggerated praise.
- Do not use placeholders.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        text = clean_text(
            getattr(response, "text", ""),
            "",
        )

        if not text:
            return None

        subject = ""
        email = ""
        dm = ""

        lines = text.splitlines()

        current = None

        for line in lines:

            stripped = line.strip()

            if stripped.upper().startswith("SUBJECT:"):
                current = "subject"
                subject = stripped.split(
                    ":", 1
                )[1].strip()

            elif stripped.upper().startswith("EMAIL:"):
                current = "email"
                email = stripped.split(
                    ":", 1
                )[1].strip()

            elif stripped.upper().startswith("DM:"):
                current = "dm"
                dm = stripped.split(
                    ":", 1
                )[1].strip()

            elif current == "subject":
                subject += " " + stripped

            elif current == "email":
                email += "\n" + stripped

            elif current == "dm":
                dm += " " + stripped

        if not subject or not email or not dm or not _message_lengths_valid(email, dm):
            log.warning(
                "Gemini returned an unexpected format for %s.",
                name,
            )
            return None

        return {
            "email_subject": subject,
            "email_pitch": email.strip(),
            "instagram_dm": dm.strip(),
            "personalization_source": "gemini",
        }

    except Exception as e:
        log.warning(
            "Gemini failed for %s: %s",
            name,
            e,
        )

        return None


def _word_count(value):
    return len(re.findall(r"\b[\w'-]+\b", value))


def _message_lengths_valid(email, dm):
    """Enforce the assignment's word-count ranges for model output."""
    email_words = _word_count(email)
    dm_words = _word_count(dm)
    return 60 <= email_words <= 105 and 15 <= dm_words <= 30


# =========================================================
# Main Message Generator
# =========================================================

def generate_outreach_message(influencer):
    """
    Try Gemini first.

    If Gemini fails, automatically use fallback.
    """

    gemini_result = generate_gemini_message(
        influencer
    )

    if gemini_result:
        return gemini_result

    return generate_fallback_message(
        influencer
    )


# =========================================================
# Run Personalization
# =========================================================

def run_personalization():
    """
    Load enriched influencers and generate messages.
    """

    log.info("=" * 55)
    log.info("  Personalization Module — starting")
    log.info("  Input : %s", INPUT_CSV)
    log.info("  Output: %s", OUTPUT_CSV)
    log.info("=" * 55)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    log.info(
        "Loaded %s influencers.",
        len(df),
    )

    email_subjects = []
    email_pitches = []
    instagram_dms = []
    sources = []

    gemini_count = 0
    fallback_count = 0

    total = len(df)

    for index, row in df.iterrows():

        influencer = row.to_dict()

        name = clean_text(
            influencer.get("name"),
            "Unknown",
        )

        log.info(
            "Processing %s/%s: %s",
            index + 1,
            total,
            name,
        )

        result = generate_outreach_message(
            influencer
        )

        email_subjects.append(
            result["email_subject"]
        )

        email_pitches.append(
            result["email_pitch"]
        )

        instagram_dms.append(
            result["instagram_dm"]
        )

        sources.append(
            result["personalization_source"]
        )

        if result["personalization_source"] == "gemini":
            gemini_count += 1
        else:
            fallback_count += 1

    df["email_subject"] = email_subjects
    df["email_pitch"] = email_pitches
    df["instagram_dm"] = instagram_dms
    df["personalization_source"] = sources

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    log.info("=" * 55)
    log.info("Personalization complete.")
    log.info("Saved: %s", OUTPUT_CSV)
    log.info("Total influencers: %s", len(df))
    log.info("Gemini messages: %s", gemini_count)
    log.info("Fallback messages: %s", fallback_count)
    log.info("=" * 55)

    return df


# =========================================================
# Standalone Test
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    result_df = run_personalization()

    print()
    print("=============================================")
    print(" Personalization Summary")
    print("=============================================")
    print(
        f"Total influencers: {len(result_df)}"
    )

    if not result_df.empty:

        first = result_df.iloc[0]

        print("\nEMAIL SUBJECT:")
        print(first["email_subject"])

        print("\nEMAIL PITCH:")
        print(first["email_pitch"])

        print("\nINSTAGRAM DM:")
        print(first["instagram_dm"])

        print("\nPERSONALIZATION SOURCE:")
        print(first["personalization_source"])
