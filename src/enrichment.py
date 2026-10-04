# src/enrichment.py
"""
Enrichment Module
==================
Enriches shortlisted micro-influencer profiles using the YouTube Data API v3
and publicly available channel information.

Responsibilities:
  - Load data/influencers_filtered.csv and keep only status=PASS rows.
  - For each channel, call the YouTube API to:
      - Fetch the upload playlist ID (channels.list with part=contentDetails).
      - Retrieve the most recent N video IDs from that playlist
        (playlistItems.list).
      - Fetch video-level statistics (videos.list with part=statistics) and
        calculate a real engagement rate:
          engagement_rate = (total_likes + total_comments) / total_views
        expressed as a percentage, rounded to 2 decimal places.
      - Extract content themes from video titles.
  - Attempt to find a public contact email from the channel description
    (the only reliably public text field available without scraping).
  - Pass through name, platform, profile_url, subscribers, niche from the
    filtered CSV without modification.
  - Save enriched records to data/influencers_enriched.csv.

Data integrity rules:
  - Never fabricate or guess data.
  - If engagement cannot be computed, store "Not Available" and log why.
  - If no email is found in public text, store exactly "Not Found".
  - Optional fields (website, audience_age, audience_gender,
    audience_geography) are stored as empty string when unavailable.

Output CSV columns:
  name, platform, profile_url, subscribers, engagement_rate, niche,
  content_themes, contact_email,
  website, audience_age, audience_gender, audience_geography,
  channel_id, enrichment_notes

Usage:
  python src/enrichment.py
  # or import in app.py:
  from src.enrichment import run_enrichment
"""

import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_CSV = Path("data/influencers_filtered.csv")
OUTPUT_CSV = Path("data/influencers_enriched.csv")

# Number of recent videos to sample for engagement rate calculation.
# Kept small (10) to minimise quota usage while still being representative.
VIDEOS_TO_SAMPLE = 10

# Pause between API calls (seconds) to be a polite API consumer.
API_PAUSE = 0.3

# Maximum number of content themes to record per channel.
MAX_THEMES = 5

# Regex to find email addresses in plain text (channel description / about).
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Columns written to the output CSV (in order).
OUTPUT_COLUMNS = [
    "name",
    "platform",
    "profile_url",
    "subscribers",
    "engagement_rate",
    "niche",
    "content_themes",
    "contact_email",
    "website",
    "audience_age",
    "audience_gender",
    "audience_geography",
    "channel_id",
    "enrichment_notes",
]

# ---------------------------------------------------------------------------
# YouTube API client (shared across all enrichment calls)
# ---------------------------------------------------------------------------


def _build_youtube_client():
    """
    Build and return an authenticated YouTube Data API v3 client.

    Reads YOUTUBE_API_KEY from the environment (.env loaded above).

    Returns:
        googleapiclient Resource for the YouTube v3 API.

    Raises:
        ValueError: If YOUTUBE_API_KEY is missing or empty.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY is not set. "
            "Add it to .env before running enrichment."
        )
    return build("youtube", "v3", developerKey=api_key)


# ---------------------------------------------------------------------------
# Step 1: Load shortlisted influencers
# ---------------------------------------------------------------------------


def load_shortlisted(input_path=INPUT_CSV):
    """
    Load influencers_filtered.csv and return only rows with status=PASS.

    Args:
        input_path: Path to the filtered CSV produced by filtering.py.

    Returns:
        pandas DataFrame containing only PASS-status rows, reset index.

    Raises:
        FileNotFoundError: If the filtered CSV does not exist.
        ValueError:        If the 'status' column is absent.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            "Filtered influencer CSV not found: {}\n"
            "Run 'python src/filtering.py' first.".format(input_path)
        )

    df = pd.read_csv(input_path, dtype={"subscribers": "Int64"})
    log.info("Loaded %d rows from %s", len(df), input_path)

    if "status" not in df.columns:
        raise ValueError(
            "Column 'status' not found in {}. "
            "Ensure filtering.py has been run correctly.".format(input_path)
        )

    shortlisted = df[df["status"] == "PASS"].reset_index(drop=True)
    log.info(
        "%d channels shortlisted (status=PASS) out of %d total.",
        len(shortlisted), len(df),
    )
    return shortlisted


# ---------------------------------------------------------------------------
# Step 2: Find contact email from public channel data
# ---------------------------------------------------------------------------


def find_contact_email(description):
    """
    Search the channel description for a publicly listed email address.

    Only the channel description is checked because it is the only text
    field reliably returned by the YouTube Data API without HTML scraping.
    No external HTTP requests are made; no email is guessed or fabricated.

    Args:
        description: Channel description string (may be empty).

    Returns:
        str: The first email address found, or "Not Found" if none.
    """
    if not description:
        return "Not Found"

    matches = _EMAIL_RE.findall(description)
    if matches:
        email = matches[0].lower()
        log.debug("    Email found: %s", email)
        return email

    return "Not Found"


# ---------------------------------------------------------------------------
# Step 3: Fetch recent video IDs from upload playlist
# ---------------------------------------------------------------------------


def _get_upload_playlist_id(youtube, channel_id):
    """
    Retrieve the uploads playlist ID for a channel via channels.list.

    The uploads playlist contains all public videos the channel has posted
    in reverse-chronological order.

    Args:
        youtube:    Authenticated YouTube API client.
        channel_id: YouTube channel ID string.

    Returns:
        str or None: The uploads playlist ID, or None on failure.
    """
    try:
        resp = (
            youtube.channels()
            .list(part="contentDetails", id=channel_id)
            .execute()
        )
        items = resp.get("items", [])
        if not items:
            log.warning("    No channel found for ID: %s", channel_id)
            return None
        playlist_id = (
            items[0]
            .get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )
        return playlist_id
    except HttpError as exc:
        log.error("    API error fetching playlist for %s: %s", channel_id, exc)
        return None


def _get_recent_video_ids(youtube, playlist_id, max_videos=VIDEOS_TO_SAMPLE):
    """
    Retrieve the most recent video IDs from an uploads playlist.

    Args:
        youtube:     Authenticated YouTube API client.
        playlist_id: The uploads playlist ID.
        max_videos:  Maximum number of video IDs to retrieve.

    Returns:
        List of video ID strings (may be empty on failure or empty playlist).
    """
    video_ids = []
    try:
        resp = (
            youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=max_videos,
            )
            .execute()
        )
        for item in resp.get("items", []):
            vid = item.get("contentDetails", {}).get("videoId")
            if vid:
                video_ids.append(vid)
    except HttpError as exc:
        log.error("    API error fetching playlist items: %s", exc)

    return video_ids


# ---------------------------------------------------------------------------
# Step 4: Calculate engagement rate from real video statistics
# ---------------------------------------------------------------------------


def calculate_engagement(youtube, channel_id):
    """
    Calculate an engagement rate for the channel from recent public video
    statistics using the YouTube Data API.

    Formula:
        engagement_rate (%) = (sum of likes + sum of comments)
                              / sum of views * 100

    Only videos with non-zero view counts are included in the calculation
    to avoid division-by-zero and to discard unpublished/private videos
    where statistics are zero.

    Args:
        youtube:    Authenticated YouTube API client.
        channel_id: YouTube channel ID string.

    Returns:
        (rate_str: str, notes: str)
        rate_str is "{float}%" rounded to 2 decimal places, or
        "Not Available" when the rate cannot be reliably computed.
        notes documents the reason when "Not Available".
    """
    # Step A: Get the uploads playlist
    playlist_id = _get_upload_playlist_id(youtube, channel_id)
    time.sleep(API_PAUSE)

    if not playlist_id:
        return "Not Available", "Could not retrieve uploads playlist"

    # Step B: Get recent video IDs
    video_ids = _get_recent_video_ids(youtube, playlist_id)
    time.sleep(API_PAUSE)

    if not video_ids:
        return "Not Available", "No public videos found in uploads playlist"

    # Step C: Fetch statistics for those videos
    try:
        stats_resp = (
            youtube.videos()
            .list(
                part="statistics,snippet",
                id=",".join(video_ids),
            )
            .execute()
        )
    except HttpError as exc:
        log.error("    API error fetching video stats: %s", exc)
        return "Not Available", "API error fetching video statistics: {}".format(exc)

    time.sleep(API_PAUSE)

    # Step D: Aggregate statistics
    total_views = 0
    total_likes = 0
    total_comments = 0
    videos_counted = 0

    for item in stats_resp.get("items", []):
        stats = item.get("statistics", {})
        views = int(stats.get("viewCount", 0) or 0)
        if views == 0:
            continue  # skip videos with no views (private/just-uploaded)
        likes = int(stats.get("likeCount", 0) or 0)
        comments = int(stats.get("commentCount", 0) or 0)
        total_views += views
        total_likes += likes
        total_comments += comments
        videos_counted += 1

    if videos_counted == 0 or total_views == 0:
        return (
            "Not Available",
            "No videos with public view counts found (sample size: {})".format(
                len(video_ids)
            ),
        )

    rate = (total_likes + total_comments) / total_views * 100
    notes = "Computed from {} recent videos ({} likes + {} comments / {} views)".format(
        videos_counted, total_likes, total_comments, total_views
    )
    log.debug("    Engagement: %.2f%% (%s)", rate, notes)
    return "{:.2f}%".format(rate), notes


# ---------------------------------------------------------------------------
# Step 5: Extract content themes from recent video titles
# ---------------------------------------------------------------------------


def extract_content_themes(youtube, channel_id, max_themes=MAX_THEMES):
    """
    Derive content themes from the titles of the channel's most recent
    public videos.

    Themes are extracted by taking the most frequent meaningful words
    (>= 4 characters, excluding common stop-words) from recent video
    titles. Only data returned by the YouTube API is used.

    Args:
        youtube:    Authenticated YouTube API client.
        channel_id: YouTube channel ID string.
        max_themes: Maximum number of theme words to return.

    Returns:
        str: Comma-separated theme keywords, or empty string on failure.
    """
    # Reuse the upload playlist lookup
    playlist_id = _get_upload_playlist_id(youtube, channel_id)
    time.sleep(API_PAUSE)

    if not playlist_id:
        return ""

    video_ids = _get_recent_video_ids(youtube, playlist_id, max_videos=20)
    time.sleep(API_PAUSE)

    if not video_ids:
        return ""

    try:
        resp = (
            youtube.videos()
            .list(part="snippet", id=",".join(video_ids))
            .execute()
        )
    except HttpError as exc:
        log.error("    API error fetching video snippets: %s", exc)
        return ""

    time.sleep(API_PAUSE)

    # Collect all titles
    titles = []
    for item in resp.get("items", []):
        title = item.get("snippet", {}).get("title", "")
        if title:
            titles.append(title.lower())

    if not titles:
        return ""

    # Tokenise and count meaningful words
    stop_words = {
        "with", "this", "that", "from", "your", "have", "will", "about",
        "what", "when", "into", "more", "some", "been", "they", "their",
        "were", "would", "could", "should", "than", "then", "also", "just",
        "like", "only", "very", "make", "made", "most", "over", "such",
        "much", "even", "come", "here", "back", "does", "need", "know",
        "full", "part", "each", "work", "good", "well", "best", "real",
        "course", "video", "tutorial", "learn", "using", "build", "step",
        "guide", "complete", "episode", "series", "2024", "2025",
    }

    word_freq = {}
    token_re = re.compile(r"[a-z]{4,}")
    for title in titles:
        for word in token_re.findall(title):
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

    if not word_freq:
        return ""

    # Sort by frequency descending, take top N
    top_words = sorted(word_freq, key=lambda w: word_freq[w], reverse=True)
    themes = top_words[:max_themes]
    return ", ".join(themes)


# ---------------------------------------------------------------------------
# Step 6: Enrich a single influencer profile
# ---------------------------------------------------------------------------


def enrich_profile(row, youtube):
    """
    Enrich one influencer row with real API-sourced data.

    Passes through: name, platform, profile_url, subscribers, niche.
    Enriches:       engagement_rate, content_themes, contact_email.
    Sets empty:     website, audience_age, audience_gender,
                    audience_geography (not publicly available via the
                    YouTube Data API without channel ownership / OAuth).

    Args:
        row:     pandas Series representing a single PASS-status influencer.
        youtube: Authenticated YouTube API client.

    Returns:
        dict suitable for a single row of the output CSV.
    """
    channel_id = str(row.get("channel_id", "")).strip()
    name = str(row.get("name", "")).strip()
    description = str(row.get("description", "")).strip()

    log.info("Enriching: %s  (channel_id=%s)", name, channel_id)

    notes_parts = []

    # --- Engagement rate ---
    if channel_id:
        engagement_rate, eng_note = calculate_engagement(youtube, channel_id)
        notes_parts.append(eng_note)
    else:
        engagement_rate = "Not Available"
        notes_parts.append("No channel_id — cannot fetch engagement")

    # --- Content themes ---
    if channel_id:
        content_themes = extract_content_themes(youtube, channel_id)
        if not content_themes:
            notes_parts.append("No content themes extracted")
    else:
        content_themes = ""
        notes_parts.append("No channel_id — cannot fetch content themes")

    # --- Contact email (from description only — no scraping) ---
    contact_email = find_contact_email(description)
    if contact_email == "Not Found":
        notes_parts.append("No public email in channel description")

    log.info(
        "  Done: engagement=%s | themes=%s | email=%s",
        engagement_rate,
        content_themes[:40] + "..." if len(content_themes) > 40 else content_themes,
        contact_email,
    )

    return {
        "name":               name,
        "platform":           str(row.get("platform", "YouTube")),
        "profile_url":        str(row.get("profile_url", "")),
        "subscribers":        row.get("subscribers"),
        "engagement_rate":    engagement_rate,
        "niche":              str(row.get("niche", "")),
        "content_themes":     content_themes,
        "contact_email":      contact_email,
        # Optional fields — not available via public API without OAuth
        "website":            "",
        "audience_age":       "",
        "audience_gender":    "",
        "audience_geography": "",
        # Carry-through for traceability
        "channel_id":         channel_id,
        "enrichment_notes":   "; ".join(filter(None, notes_parts)),
    }


# ---------------------------------------------------------------------------
# Step 7: Run the full enrichment pipeline
# ---------------------------------------------------------------------------


def run_enrichment(input_path=INPUT_CSV, output_path=OUTPUT_CSV):
    """
    Execute the full enrichment pipeline.

    Steps:
      1. Load data/influencers_filtered.csv; keep only status=PASS rows.
      2. Build the YouTube API client.
      3. For each shortlisted channel, call enrich_profile().
      4. Save all enriched records to data/influencers_enriched.csv.

    Args:
        input_path:  Path to the filtered CSV (default: data/influencers_filtered.csv).
        output_path: Path for the enriched output CSV
                     (default: data/influencers_enriched.csv).

    Returns:
        pandas DataFrame of all enriched records.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError:        If required columns are absent or API key missing.
    """
    log.info("=" * 55)
    log.info("  Enrichment -- starting")
    log.info("  Input : %s", input_path)
    log.info("  Output: %s", output_path)
    log.info("=" * 55)

    # Step 1: Load shortlisted influencers
    shortlisted = load_shortlisted(input_path)

    if shortlisted.empty:
        log.warning("No PASS-status influencers to enrich. Exiting.")
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        save_results(empty_df, output_path)
        return empty_df

    # Step 2: Build YouTube API client
    youtube = _build_youtube_client()

    # Step 3: Enrich each profile
    enriched_records = []
    total = len(shortlisted)

    for idx, row in shortlisted.iterrows():
        log.info("[%d/%d] Processing: %s", idx + 1, total, row.get("name", "?"))
        try:
            record = enrich_profile(row, youtube)
            enriched_records.append(record)
        except Exception as exc:
            # One failed channel must not abort the entire run
            log.error(
                "  Unexpected error enriching '%s': %s — skipping.",
                row.get("name", "?"), exc,
            )
            enriched_records.append({
                "name":               str(row.get("name", "")),
                "platform":           str(row.get("platform", "YouTube")),
                "profile_url":        str(row.get("profile_url", "")),
                "subscribers":        row.get("subscribers"),
                "engagement_rate":    "Not Available",
                "niche":              str(row.get("niche", "")),
                "content_themes":     "",
                "contact_email":      "Not Found",
                "website":            "",
                "audience_age":       "",
                "audience_gender":    "",
                "audience_geography": "",
                "channel_id":         str(row.get("channel_id", "")),
                "enrichment_notes":   "Error: {}".format(exc),
            })

        time.sleep(API_PAUSE)  # brief pause between channels

    # Step 4: Save
    result_df = pd.DataFrame(enriched_records, columns=OUTPUT_COLUMNS)
    save_results(result_df, output_path)

    # Summary
    found_email = int((result_df["contact_email"] != "Not Found").sum())
    has_engagement = int((result_df["engagement_rate"] != "Not Available").sum())
    print()
    print("=" * 50)
    print("  Enrichment Summary")
    print("=" * 50)
    print("  Channels enriched  : {:>5}".format(len(result_df)))
    print("  Engagement computed: {:>5}".format(has_engagement))
    print("  Emails found       : {:>5}".format(found_email))
    print("=" * 50)
    print()

    log.info("Enrichment complete. %d records saved to %s", len(result_df), output_path)
    return result_df


# ---------------------------------------------------------------------------
# Step 8: Save results
# ---------------------------------------------------------------------------


def save_results(df, output_path=OUTPUT_CSV):
    """
    Write the enriched DataFrame to a CSV file.

    Creates the parent directory if it does not already exist.

    Args:
        df:          DataFrame to save.
        output_path: Destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Saved %d rows to %s", len(df), output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        result = run_enrichment()

        if not result.empty:
            pd.set_option("display.max_colwidth", 50)
            print("Enriched profiles:")
            print(
                result[["name", "subscribers", "engagement_rate", "contact_email"]]
                .to_string(index=False)
            )

        sys.exit(0)

    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    except ValueError as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)
