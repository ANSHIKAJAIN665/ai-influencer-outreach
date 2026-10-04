# src/filtering.py
"""
Filtering & Classification Module
===================================
Reads raw discovered channels from data/influencers_raw.csv (produced by
discovery.py) and applies micro-influencer criteria plus a Technology/AI
niche relevance check.

Responsibilities:
  - Load data/influencers_raw.csv.
  - Apply subscriber range filter: 5,000 <= subscribers <= 100,000.
  - Detect Technology / AI niche relevance from channel name and description.
  - Classify each channel as PASS or FAIL with a clear failure_reason.
  - Assign a niche label and content_relevance value.
  - Save the fully-classified dataset to data/influencers_filtered.csv.
  - Print a concise summary to stdout.

Output CSV columns:
  name, channel_id, platform, profile_url, subscribers, description,
  niche, content_relevance, status, failure_reason

Usage:
  python src/filtering.py
  # or import in app.py:
  from src.filtering import run_filtering
"""

import logging
import sys
from pathlib import Path

import pandas as pd

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

INPUT_CSV = Path("data/influencers_raw.csv")
OUTPUT_CSV = Path("data/influencers_filtered.csv")

# Subscriber range for micro-influencers (both inclusive)
MIN_SUBSCRIBERS = 5_000
MAX_SUBSCRIBERS = 100_000

# Expected columns from discovery.py output
REQUIRED_COLUMNS = {"name", "channel_id", "platform", "profile_url", "subscribers", "description"}

# Output column order
OUTPUT_COLUMNS = [
    "name",
    "channel_id",
    "platform",
    "profile_url",
    "subscribers",
    "description",
    "niche",
    "content_relevance",
    "status",
    "failure_reason",
]

# ---------------------------------------------------------------------------
# Niche keyword mapping
# ---------------------------------------------------------------------------
# Each entry maps a niche label to a list of keyword patterns.
# Matching rules:
#   - All matching is case-insensitive.
#   - Patterns that are plain words / phrases are matched with \b word
#     boundaries so that short tokens ("ml", "ai", "sql") do not fire
#     inside longer words ("html", "email", "visual").
#   - Priority order is intentional: more specific niches (Data Science,
#     Machine Learning, Programming, Software Development, Technology) are
#     checked BEFORE Artificial Intelligence so that a channel clearly about
#     Python programming is not mislabelled AI just because its description
#     mentions one general AI term.
# ---------------------------------------------------------------------------

# Each tuple is (pattern_string, use_word_boundary).
# use_word_boundary=True  -> wrapped in \b...\b for whole-word matching
# use_word_boundary=False -> matched as a substring (for multi-word phrases
#                            that already contain spaces, punctuation, etc.)

NICHE_KEYWORDS = {
    "Data Science": [
        ("data science", False),
        ("data analysis", False),
        ("data analytics", False),
        ("data engineering", False),
        ("pandas", True),
        ("numpy", True),
        ("matplotlib", True),
        ("sql", True),
        ("tableau", True),
        ("power bi", False),
        ("statistics", True),
        ("kaggle", True),
    ],
    "Machine Learning": [
        ("machine learning", False),
        ("ml", True),
        ("scikit", True),
        ("scikit-learn", False),
        ("tensorflow", True),
        ("pytorch", True),
        ("keras", True),
        ("xgboost", True),
        ("reinforcement learning", False),
        ("supervised learning", False),
        ("unsupervised learning", False),
    ],
    "Programming": [
        ("python", True),
        ("javascript", True),
        ("typescript", True),
        ("java", True),
        ("c\\+\\+", True),
        ("rust", True),
        ("golang", True),
        ("coding", True),
        ("programming", True),
        ("web development", False),
        ("backend", True),
        ("frontend", True),
        ("full stack", False),
        ("fullstack", True),
    ],
    "Software Development": [
        ("software development", False),
        ("software engineering", False),
        ("devops", True),
        ("cloud computing", False),
        ("kubernetes", True),
        ("docker", True),
        ("aws", True),
        ("azure", True),
        ("gcp", True),
        ("open source", False),
        ("system design", False),
    ],
    "Technology": [
        ("technology", True),
        ("tech", True),
        ("robotics", True),
        ("computer science", False),
        ("cybersecurity", True),
        ("blockchain", True),
        ("augmented reality", False),
        ("virtual reality", False),
        ("gadget", True),
        ("hardware", True),
        ("semiconductor", True),
        ("electronics", True),
    ],
    "Artificial Intelligence": [
        ("artificial intelligence", False),
        ("generative ai", False),
        ("llm", True),
        ("large language model", False),
        ("gpt", True),
        ("chatgpt", True),
        ("openai", True),
        ("ai", True),
        ("deep learning", False),
        ("neural network", False),
        ("computer vision", False),
        ("nlp", True),
        ("natural language processing", False),
    ],
}


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------


def load_data(input_path=INPUT_CSV):
    """
    Load the raw influencer CSV produced by discovery.py.

    Validates that the file exists and contains the expected columns.

    Args:
        input_path: Path to the raw CSV file.

    Returns:
        pandas DataFrame with at least the columns in REQUIRED_COLUMNS.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError:        If required columns are missing from the file.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            "Raw influencer CSV not found: {}\n"
            "Run 'python src/discovery.py' first to generate it.".format(input_path)
        )

    df = pd.read_csv(input_path, dtype={"subscribers": "Int64"})
    log.info("Loaded %d rows from %s", len(df), input_path)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Input CSV is missing required columns: {}\n"
            "Found columns: {}".format(missing_cols, list(df.columns))
        )

    # Normalise text columns: fill NaN with empty strings
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# Step 2: Subscriber range check
# ---------------------------------------------------------------------------


def check_subscriber_range(subscribers):
    """
    Check whether a subscriber count falls within the micro-influencer range.

    Args:
        subscribers: Raw subscriber value (int, float, or pandas NA).

    Returns:
        (passes: bool, reason: str)
        reason is an empty string when passes is True.
    """
    if pd.isna(subscribers):
        return False, "Missing subscriber count"

    try:
        subs = int(subscribers)
    except (ValueError, TypeError):
        return False, "Invalid subscriber count"

    if subs < MIN_SUBSCRIBERS:
        return False, "Below {:,} subscribers".format(MIN_SUBSCRIBERS)
    if subs > MAX_SUBSCRIBERS:
        return False, "Above {:,} subscribers".format(MAX_SUBSCRIBERS)

    return True, ""


# ---------------------------------------------------------------------------
# Step 3: Niche classification
# ---------------------------------------------------------------------------

import re as _re

# Pre-compile one regex per (pattern, use_word_boundary) pair at module load
# to avoid repeated compilation inside the hot per-row path.
_COMPILED_NICHE_PATTERNS = {
    niche_label: [
        _re.compile(
            r"\b" + pat + r"\b" if use_wb else _re.escape(pat),
            _re.IGNORECASE,
        )
        for pat, use_wb in patterns
    ]
    for niche_label, patterns in NICHE_KEYWORDS.items()
}


def classify_niche(name, description):
    """
    Determine whether the channel is relevant to Technology/AI and assign
    a specific niche label.

    Uses pre-compiled regular expressions with word-boundary (\\b) anchors
    for single-token patterns so that short tokens like "ml", "ai", "sql"
    do not match inside longer words such as "html", "email", "visual".
    Multi-word phrases (e.g. "machine learning") are matched as substrings
    because they are inherently unambiguous.

    Only data already present in the CSV is used -- no external look-ups.
    Classification is deterministic: given the same name + description, the
    same niche is always returned.

    Priority order (first match wins):
      Data Science -> Machine Learning -> Programming ->
      Software Development -> Technology -> Artificial Intelligence

    Args:
        name:        Channel display name (str).
        description: Channel description text (str).

    Returns:
        (niche: str, content_relevance: str)
        niche is an empty string when not relevant.
        content_relevance is "Relevant" or "Not Relevant".
    """
    combined = name + " " + description  # case handled by IGNORECASE flag

    for niche_label, compiled_patterns in _COMPILED_NICHE_PATTERNS.items():
        for pattern in compiled_patterns:
            if pattern.search(combined):
                return niche_label, "Relevant"

    return "", "Not Relevant"


# ---------------------------------------------------------------------------
# Step 4: Classify a single influencer row
# ---------------------------------------------------------------------------


def classify_influencer(row):
    """
    Apply all classification rules to a single DataFrame row and return
    the row augmented with the four classification columns.

    Classification logic:
      PASS  -> subscribers in [5,000 - 100,000] AND relevant to Technology/AI
      FAIL  -> any criterion not met

    Args:
        row: A pandas Series representing one channel record.

    Returns:
        The same Series with niche, content_relevance, status, and
        failure_reason added.
    """
    # --- Subscriber range ---
    subs_ok, subs_reason = check_subscriber_range(row.get("subscribers"))

    # --- Niche relevance ---
    niche, content_relevance = classify_niche(
        str(row.get("name", "")),
        str(row.get("description", "")),
    )
    relevance_ok = content_relevance == "Relevant"

    # --- Aggregate failure reasons ---
    reasons = []
    if not subs_ok:
        reasons.append(subs_reason)
    if not relevance_ok:
        reasons.append("Not relevant to Technology/AI")

    if not reasons:
        status = "PASS"
        failure_reason = ""
    else:
        status = "FAIL"
        failure_reason = "; ".join(reasons)

    row = row.copy()
    row["niche"] = niche if niche else ""
    row["content_relevance"] = content_relevance
    row["status"] = status
    row["failure_reason"] = failure_reason

    return row


# ---------------------------------------------------------------------------
# Step 5: Run the full filtering pipeline
# ---------------------------------------------------------------------------


def run_filtering(input_path=INPUT_CSV, output_path=OUTPUT_CSV):
    """
    Execute the complete filtering and classification pipeline.

    Steps:
      1. Load data/influencers_raw.csv.
      2. Apply classify_influencer() to every row.
      3. Save the fully-classified DataFrame to data/influencers_filtered.csv.
      4. Print a summary to stdout.

    Args:
        input_path:  Path to the raw CSV (default: data/influencers_raw.csv).
        output_path: Path to write the filtered CSV
                     (default: data/influencers_filtered.csv).

    Returns:
        pandas DataFrame of all classified channels (PASS and FAIL).

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError:        If required columns are absent.
    """
    log.info("=" * 55)
    log.info("  Filtering & Classification -- starting")
    log.info("  Subscriber range: %d - %d", MIN_SUBSCRIBERS, MAX_SUBSCRIBERS)
    log.info("  Input : %s", input_path)
    log.info("  Output: %s", output_path)
    log.info("=" * 55)

    # Step 1: Load
    df = load_data(input_path)
    total_raw = len(df)

    if total_raw == 0:
        log.warning("Input CSV is empty -- nothing to classify.")
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        save_results(empty_df, output_path)
        _print_summary(0, 0, 0)
        return empty_df

    # Step 2: Classify every row
    log.info("Classifying %d channels ...", total_raw)
    classified_rows = df.apply(classify_influencer, axis=1)
    result_df = pd.DataFrame(classified_rows)

    # Ensure all output columns exist in correct order
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = ""
    result_df = result_df[OUTPUT_COLUMNS]

    # Step 3: Save
    save_results(result_df, output_path)

    # Step 4: Summary
    n_pass = int((result_df["status"] == "PASS").sum())
    n_fail = int((result_df["status"] == "FAIL").sum())
    _print_summary(total_raw, n_pass, n_fail)

    log.info("Filtering complete.")
    return result_df


# ---------------------------------------------------------------------------
# Step 6: Save results
# ---------------------------------------------------------------------------


def save_results(df, output_path=OUTPUT_CSV):
    """
    Write the classified DataFrame to a CSV file.

    Creates the parent directory if it does not already exist.

    Args:
        df:          DataFrame to save (should contain OUTPUT_COLUMNS).
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Saved %d rows to %s", len(df), output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_summary(total, n_pass, n_fail):
    """Print a human-readable filtering summary to stdout."""
    pass_pct = (n_pass / total * 100) if total > 0 else 0.0
    print()
    print("=" * 45)
    print("  Filtering Summary")
    print("=" * 45)
    print("  Total raw channels : {:>6}".format(total))
    print("  Passed             : {:>6}  ({:.1f}%)".format(n_pass, pass_pct))
    print("  Failed             : {:>6}".format(n_fail))
    print("=" * 45)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        result = run_filtering()
        passed = result[result["status"] == "PASS"]

        if not passed.empty:
            print("Channels that PASSED all filters:")
            pd.set_option("display.max_colwidth", 55)
            print(
                passed[["name", "subscribers", "niche", "profile_url"]]
                .to_string(index=False)
            )
        else:
            print("No channels passed all filters.")
            print("Check {} for failure reasons.".format(OUTPUT_CSV))

        sys.exit(0)

    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    except ValueError as exc:
        log.error("Data error: %s", exc)
        sys.exit(1)
