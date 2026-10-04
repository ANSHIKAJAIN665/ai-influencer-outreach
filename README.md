# Automated Micro-Influencer Outreach System

An end-to-end Python prototype for discovering, qualifying, enriching, personalizing, and safely delivering outreach to **Technology / AI YouTube micro-influencers**. It was built for the EDXSO AI Engineer Intern Assignment 1.

The repository includes an existing test run with **269 real YouTube channels** and **60 shortlisted micro-influencers**. Data comes from the YouTube Data API; unavailable fields are explicitly recorded as `Not Found` or `Not Available`.

## Workflow

```text
YouTube Data API → discovery → classification → enrichment → personalized messages
                                                              ↓
                                             SMTP simulation / delivery → SQLite tracker
```

| Stage | Module | Result |
|---|---|---|
| Discovery | `src/discovery.py` | Searches AI/tech queries, deduplicates channels, fetches public metadata. |
| Filtering | `src/filtering.py` | Labels every profile `PASS` or `FAIL` with a reason. |
| Enrichment | `src/enrichment.py` | Uses recent public video statistics to calculate engagement and themes; extracts only publicly listed emails. |
| Personalization | `src/personalization.py` | Produces a 60–90-word email and 15–30-word DM from actual profile signals. Gemini is optional; a deterministic local fallback keeps the system usable. |
| Sending | `src/sender.py` | Simulates by default, or sends through SMTP when explicitly enabled. A SQLite log prevents duplicate sends. |

## Dataset outputs

All generated data is stored in `data/` and ignored by Git because profile/contact data may be sensitive.

| File | Contents |
|---|---|
| `influencers_raw.csv` | Every discovered channel and public description. |
| `influencers_filtered.csv` | Complete `PASS`/`FAIL` classification plus `failure_reason`. |
| `influencers_enriched.csv` | Shortlisted profiles with engagement, themes, public email status, and data-source notes. |
| `influencers_personalized.csv` | Qualified influencers plus email subject, email pitch, Instagram/manual DM, and generation source. |
| `outreach_tracker.csv` | Current run’s attempted delivery events. |
| `outreach.db` | Persistent duplicate-protection and outreach audit log. |

## Filtering logic

A channel passes only if it:

- has **5,000–100,000** public subscribers;
- is relevant to Technology/AI, based on deterministic matching of its channel name and public description.

Profiles that fail retain an explicit reason such as `Below 5,000 subscribers`, `Above 100,000 subscribers`, or `Not relevant to Technology/AI`. The classifier assigns a focused niche, including Data Science, Machine Learning, Programming, Software Development, Technology, or Artificial Intelligence.

## Enrichment integrity

Engagement is calculated from a sample of up to 10 recent public videos:

`(total likes + total comments) / total views × 100`

Content themes are extracted from recent public video titles. Contact email is parsed only from the returned public channel description—never guessed. Audience age, gender, and geography are left empty because they are not available through the public API. Instagram DMs are generated for **manual review/sending**, not automated, to respect platform restrictions.

## Setup

1. Create a virtual environment and install requirements:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add a YouTube Data API v3 key. Enable **YouTube Data API v3** in Google Cloud.

3. Keep `DRY_RUN=true` to safely demonstrate delivery. Optionally configure Gemini and SMTP variables.

4. Run the complete workflow:

   ```powershell
   python app.py
   ```

Run an individual stage with `python src/filtering.py`, `python src/enrichment.py`, `python src/personalization.py`, or `python src/sender.py`.

## Sending behavior

`DRY_RUN=true` is the default: the system records a **simulated** send without sending email. When `DRY_RUN=false`, it requires `SENDER_EMAIL` and `SENDER_PASSWORD` and uses STARTTLS SMTP. Before every delivery it checks the contact email in `data/outreach.db`; addresses already logged as `sent` or `simulated` are skipped, including duplicate channels owned by the same creator. Failed messages are logged and may be retried after correcting the issue.

## Limitations and production next steps

- Public YouTube data does not provide verified audience demographics, direct email access, or a guaranteed view of all business-contact details.
- Recent-video engagement is a transparent sample, not a platform-provided channel-wide metric.
- The system is deliberately rate-limited and API-key based; scale beyond hundreds of channels with queues, retries, quota monitoring, and a managed database.
- Add human approval before any non-dry-run campaign; consent, unsubscribe handling, and applicable email/privacy rules remain the sender’s responsibility.
