#!/usr/bin/env python3
"""
backfill_espn.py — Scrape ESPNcricinfo historical data 2014–present.

This script performs a one-shot historical backfill of match data from
ESPNcricinfo.  It uses :func:`~src.scrapers.espn_historical.discover_matches`
to walk through the ESPN records pages year-by-year and format-by-format,
then calls :func:`~src.scrapers.espn_scorecard.parse_scorecard_to_db` for
each discovered match to fetch and persist the full scorecard.

The script is fully **idempotent**: already-processed match IDs are tracked
in ``data/scrape_progress.json`` and skipped on re-runs.

Usage
-----
.. code-block:: shell

    # Scrape all formats 2014–2026 (default)
    python scripts/backfill_espn.py

    # Scrape a specific year range
    python scripts/backfill_espn.py --start-year 2020 --end-year 2024

    # Scrape only T20Is and ODIs from 2022
    python scripts/backfill_espn.py --start-year 2022 --end-year 2022 \\
        --formats twenty20-internationals,one-day-internationals

    # Preview URLs without making any requests
    python scripts/backfill_espn.py --start-year 2023 --end-year 2023 --dry-run

Exit Codes
----------
0  — Completed successfully (even if some matches errored; errors are logged).
1  — Fatal initialisation error (e.g. database cannot be reached).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when invoked as a script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Imports (project-internal, after path fix)
# ---------------------------------------------------------------------------
from src.data.db import SessionLocal, init_db
from src.scrapers.espn_historical import (
    DEFAULT_FORMATS,
    RECORDS_URL_TEMPLATE,
    discover_matches,
)
from src.scrapers.espn_scorecard import parse_scorecard_to_db

# rich is in requirements.txt
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """Configure root logger, using RichHandler when rich is available."""
    level = logging.DEBUG if verbose else logging.INFO

    if _RICH_AVAILABLE:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


logger = logging.getLogger("backfill_espn")

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_espn.py",
        description=(
            "Scrape ESPNcricinfo historical match data and persist it to "
            "the Cricket Analytics database."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2014,
        metavar="YEAR",
        help="First year to scrape (inclusive, default: 2014).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        metavar="YEAR",
        help="Last year to scrape (inclusive, default: 2026).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=None,
        metavar="FMT1,FMT2",
        help=(
            "Comma-separated list of ESPN format slugs to scrape. "
            "Valid values: test-matches, one-day-internationals, "
            "twenty20-internationals, ipl. "
            "Defaults to all three international formats."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the URLs that would be scraped without making any network "
            "requests or writing to the database."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Dry-run logic
# ---------------------------------------------------------------------------

def _dry_run(start_year: int, end_year: int, formats: List[str]) -> None:
    """Print the ESPN records URLs that would be scraped, then exit."""
    if _RICH_AVAILABLE:
        console = Console()
        console.print("\n[bold cyan]Dry-run mode — no requests will be made[/bold cyan]\n")
    else:
        print("\nDry-run mode — no requests will be made\n")

    total = 0
    for year in range(start_year, end_year + 1):
        for fmt in formats:
            url = RECORDS_URL_TEMPLATE.format(year=year, match_type=fmt)
            if _RICH_AVAILABLE:
                console = Console()
                console.print(f"  [green]{url}[/green]")
            else:
                print(f"  {url}")
            total += 1

    print(f"\nTotal URLs: {total}")


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

def _run_backfill(
    start_year: int,
    end_year: int,
    formats: List[str],
) -> None:
    """
    Execute the full historical backfill.

    Iterates through every match yielded by :func:`discover_matches` and calls
    :func:`parse_scorecard_to_db` for each one.  Statistics are accumulated
    and printed on completion.
    """
    logger.info(
        "Starting backfill: years=%d–%d formats=%s",
        start_year,
        end_year,
        formats,
    )

    # Initialise DB (creates tables if needed)
    try:
        init_db()
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        sys.exit(1)

    counters = {
        "discovered": 0,
        "new_saved": 0,
        "errors": 0,
        "skipped": 0,
    }

    start_time = time.monotonic()

    if _RICH_AVAILABLE:
        _run_with_rich_progress(start_year, end_year, formats, counters)
    else:
        _run_plain(start_year, end_year, formats, counters)

    elapsed = time.monotonic() - start_time
    _print_summary(counters, elapsed)


def _run_with_rich_progress(
    start_year: int,
    end_year: int,
    formats: List[str],
    counters: dict,
) -> None:
    """Run the backfill loop with a rich progress bar."""
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        discover_task = progress.add_task(
            "Discovering matches…", total=None
        )
        scorecard_task = progress.add_task(
            "Saving scorecards…", total=None, visible=False
        )

        session = SessionLocal()
        try:
            for match in discover_matches(
                start_year=start_year, end_year=end_year, formats=formats
            ):
                counters["discovered"] += 1
                progress.update(
                    discover_task,
                    description=(
                        f"Discovered: {counters['discovered']}  "
                        f"({match['team_a']} vs {match['team_b']} "
                        f"{match['match_date'][:4]})"
                    ),
                    advance=1,
                )

                match_id = match["espn_match_id"]
                progress.update(
                    scorecard_task,
                    description=f"Saving match {match_id}…",
                    visible=True,
                    advance=0,
                )

                ok = _save_match(match_id, session)
                if ok:
                    counters["new_saved"] += 1
                    session.commit()
                    progress.update(
                        scorecard_task,
                        description=(
                            f"Saved: {counters['new_saved']}  "
                            f"Errors: {counters['errors']}"
                        ),
                        advance=1,
                    )
                else:
                    counters["errors"] += 1
        finally:
            session.close()


def _run_plain(
    start_year: int,
    end_year: int,
    formats: List[str],
    counters: dict,
) -> None:
    """Run the backfill loop with plain logging (no rich)."""
    session = SessionLocal()
    try:
        for match in discover_matches(
            start_year=start_year, end_year=end_year, formats=formats
        ):
            counters["discovered"] += 1
            match_id = match["espn_match_id"]

            if counters["discovered"] % 10 == 0:
                logger.info(
                    "Progress — discovered=%d saved=%d errors=%d",
                    counters["discovered"],
                    counters["new_saved"],
                    counters["errors"],
                )

            ok = _save_match(match_id, session)
            if ok:
                counters["new_saved"] += 1
                session.commit()
            else:
                counters["errors"] += 1
    finally:
        session.close()


def _save_match(espn_match_id: int, session) -> bool:
    """
    Call :func:`parse_scorecard_to_db` and handle any unexpected exception.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        return parse_scorecard_to_db(espn_match_id, session)
    except Exception as exc:
        logger.error(
            "Unexpected error saving match %d: %s", espn_match_id, exc, exc_info=True
        )
        try:
            session.rollback()
        except Exception:
            pass
        return False


def _print_summary(counters: dict, elapsed: float) -> None:
    """Log / print the final backfill summary."""
    summary = (
        f"\n{'=' * 55}\n"
        f"  Backfill complete in {elapsed:.1f}s\n"
        f"  Matches discovered : {counters['discovered']}\n"
        f"  New matches saved  : {counters['new_saved']}\n"
        f"  Errors             : {counters['errors']}\n"
        f"{'=' * 55}"
    )
    if _RICH_AVAILABLE:
        console = Console()
        console.print(f"\n[bold green]{summary}[/bold green]")
    else:
        print(summary)
    logger.info("Summary: %s", counters)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and dispatch to dry-run or live backfill."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    # Parse formats
    if args.formats:
        formats = [f.strip() for f in args.formats.split(",") if f.strip()]
        # Validate
        valid = set(DEFAULT_FORMATS) | {"ipl"}
        unknown = set(formats) - valid
        if unknown:
            logger.error(
                "Unknown format(s): %s. Valid options: %s",
                unknown,
                sorted(valid),
            )
            sys.exit(1)
    else:
        formats = list(DEFAULT_FORMATS)

    # Validate year range
    if args.start_year > args.end_year:
        logger.error(
            "--start-year (%d) must be <= --end-year (%d)",
            args.start_year,
            args.end_year,
        )
        sys.exit(1)

    if args.dry_run:
        _dry_run(args.start_year, args.end_year, formats)
        return

    _run_backfill(args.start_year, args.end_year, formats)


if __name__ == "__main__":
    main()
