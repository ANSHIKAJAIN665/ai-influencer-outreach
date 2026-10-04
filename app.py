"""
AI-Powered Micro-Influencer Outreach System
============================================
Main orchestration script for the complete pipeline.

Pipeline:
  Stage 1 - Discovery
  Stage 2 - Filtering
  Stage 3 - Enrichment
  Stage 4 - Personalization
  Stage 5 - Outreach
"""

import sys
import logging

from src.discovery import run_discovery
from src.filtering import run_filtering
from src.enrichment import run_enrichment
from src.personalization import run_personalization
from src.sender import run_sender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


def main():

    log.info("=" * 60)
    log.info("Starting AI Influencer Outreach Pipeline")
    log.info("=" * 60)

    # ============================================================
    # Stage 1: Discovery
    # ============================================================
    log.info("--- Stage 1: Discovery ---")

    raw_df = run_discovery()

    log.info(
        f"Discovery complete: {len(raw_df)} raw channels found."
    )

    # ============================================================
    # Stage 2: Filtering
    # ============================================================
    log.info("--- Stage 2: Filtering ---")

    filtered_df = run_filtering()

    log.info(
        f"Filtering complete: {(filtered_df['status'] == 'PASS').sum()} qualified influencers."
    )

    # ============================================================
    # Stage 3: Enrichment
    # ============================================================
    log.info("--- Stage 3: Enrichment ---")

    enriched_df = run_enrichment()

    log.info(
        f"Enrichment complete: {len(enriched_df)} influencers enriched."
    )

    # ============================================================
    # Stage 4: Personalization
    # ============================================================
    log.info("--- Stage 4: Personalization ---")

    personalized_df = run_personalization()

    log.info(
        f"Personalization complete: "
        f"{len(personalized_df)} influencers personalized."
    )

    # ============================================================
    # Stage 5: Outreach
    # ============================================================
    log.info("--- Stage 5: Outreach ---")

    sender_result = run_sender()

    log.info(
        f"Outreach stage complete: {sender_result}"
    )

    # ============================================================
    # Final Summary
    # ============================================================
    log.info("=" * 60)
    log.info("AI Influencer Outreach Pipeline COMPLETE")
    log.info("=" * 60)

    print("\n=============================================")
    print("       PIPELINE COMPLETED SUCCESSFULLY")
    print("=============================================")
    print(f"Raw channels discovered : {len(raw_df)}")
    print(f"Qualified influencers   : {(filtered_df['status'] == 'PASS').sum()}")
    print(f"Enriched influencers    : {len(enriched_df)}")
    print(f"Personalized influencers: {len(personalized_df)}")
    print(
    f"Outreach                : "
    f"{sender_result['successful']} sent, "
    f"{sender_result['failed']} failed, "
    f"{sender_result.get('skipped', 0)} skipped"
    )
    print("=============================================")


if __name__ == "__main__":

    try:
        main()

    except ValueError as e:
        log.error(f"Configuration error: {e}")
        sys.exit(1)

    except RuntimeError as e:
        log.error(f"Pipeline error: {e}")
        sys.exit(1)

    except FileNotFoundError as e:
        log.error(f"Required file not found: {e}")
        sys.exit(1)

    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        sys.exit(1)
