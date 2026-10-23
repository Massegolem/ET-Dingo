#!/usr/bin/env python3
"""
Generate per-event dingo_pipe .ini files from a NAME.ini template and a
selected_events.txt list.

For every event name found in selected_events.txt, this script takes
NAME.ini, replaces every occurrence of the literal string "NAME" with the
event name, and writes the result to <event_name>.ini in the output
directory.

Usage:
    python generate_event_inis.py \
        --template NAME.ini \
        --events selected_events.txt \
        --outdir ./inis
"""

import argparse
from pathlib import Path


def read_event_names(events_path: Path) -> list[str]:
    """Read event names from selected_events.txt.

    Skips blank lines and lines starting with '#' (used here as section
    headers like '# O4a', '# O3b', etc.).
    """
    events = []
    with events_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            events.append(line)
    return events


def generate_inis(template_path: Path, events_path: Path, outdir: Path) -> list[Path]:
    template_text = template_path.read_text()
    events = read_event_names(events_path)

    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for event in events:
        event_text = template_text.replace("NAME", event)
        out_path = outdir / f"{event}.ini"
        out_path.write_text(event_text)
        written.append(out_path)

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path, default=Path("NAME.ini"),
        help="Path to the template .ini file (default: NAME.ini)",
    )
    parser.add_argument(
        "--events", type=Path, default=Path("selected_events.txt"),
        help="Path to the event list file (default: selected_events.txt)",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("."),
        help="Directory to write the generated .ini files into (default: current dir)",
    )
    args = parser.parse_args()

    written = generate_inis(args.template, args.events, args.outdir)

    print(f"Generated {len(written)} .ini files in {args.outdir.resolve()}:")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()