#!/usr/bin/env python3
"""
filter_events.py

Filter a user-selected list of GW event names out of the full GWOSC
event-versions catalogue (event-versions.csv), keeping the latest
catalogue version of each event.

Usage:
    python filter_events.py --catalogue event-versions.csv \
                             --events selected_events.txt \
                             --out filtered_events.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def load_selected_events(path: Path) -> list[str]:
    """
    Read one event name per line from a text file, stripping
    whitespace, footnote markers (e.g. the '†' used in some tables),
    and blank lines / comments (lines starting with '#').
    """
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            # Strip footnote markers / trailing symbols some tables use
            name = re.sub(r"[†‡*]+$", "", name).strip()
            events.append(name)
    return events


def load_catalogue(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def latest_version_per_event(rows: list[dict]) -> dict[str, dict]:
    """
    Collapse the catalogue to one row per event 'name', keeping the
    row with the highest 'version' (GWTC-5.0 etc. supersede earlier
    marginal/preliminary entries for the same event).
    """
    best: dict[str, dict] = {}
    for row in rows:
        name = row["name"]
        version = int(row["version"]) if row.get("version") else 0
        if name not in best or version > int(best[name]["version"]):
            best[name] = row
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Filter selected GW events from the full GWOSC catalogue."
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=Path("event-versions.csv"),
        help="Path to the full catalogue CSV (default: event-versions.csv)",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("selected_events.txt"),
        help="Text file with one event name per line (default: selected_events.txt)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("filtered_events.csv"),
        help="Output CSV path (default: filtered_events.csv)",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Keep all catalogue versions of each selected event, "
        "instead of only the latest.",
    )
    args = parser.parse_args()

    selected = load_selected_events(args.events)
    catalogue_rows = load_catalogue(args.catalogue)

    if args.all_versions:
        matches = [row for row in catalogue_rows if row["name"] in selected]
    else:
        latest = latest_version_per_event(catalogue_rows)
        matches = [latest[name] for name in selected if name in latest]

    found_names = {row["name"] for row in matches}
    missing = [name for name in selected if name not in found_names]

    if not matches:
        print("No matching events found in the catalogue.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(matches[0].keys())
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)

    print(f"Selected events requested : {len(selected)}")
    print(f"Matched in catalogue      : {len(matches)}")
    print(f"Written to                : {args.out}")
    if missing:
        print(f"\nWARNING: {len(missing)} event(s) not found in catalogue:")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
