#!/usr/bin/env python3
"""
Generate per-mass-range train_settings_<range>.yaml and <range>.sh files from
NAME templates and a mass_range.txt list.

For every mass-range label found in mass_range.txt (e.g. "50_300"), this
script takes train_settings_NAME.yaml and NAME.sh, replaces every occurrence
of the literal string "NAME" with the mass-range label, and writes the
results to train_settings_<label>.yaml and <label>.sh in the output
directory.

Usage:
    python generate_mass_range_files.py \
        --yaml-template train_settings_NAME.yaml \
        --sh-template NAME.sh \
        --ranges mass_range.txt \
        --outdir ./mass_range_jobs
"""

import argparse
from pathlib import Path


def read_labels(ranges_path: Path) -> list[str]:
    """Read mass-range labels from mass_range.txt.

    Skips blank lines and lines starting with '#' (in case section headers
    or comments are ever added).
    """
    labels = []
    with ranges_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            labels.append(line)
    return labels


def generate_files(
    yaml_template_path: Path,
    sh_template_path: Path,
    ranges_path: Path,
    outdir: Path,
) -> list[Path]:
    yaml_text = yaml_template_path.read_text()
    sh_text = sh_template_path.read_text()
    labels = read_labels(ranges_path)

    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for label in labels:
        yaml_out = outdir / f"train_settings_{label}.yaml"
        sh_out = outdir / f"{label}.sh"

        yaml_out.write_text(yaml_text.replace("NAME", label))
        sh_out.write_text(sh_text.replace("NAME", label))

        written.extend([yaml_out, sh_out])

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml-template", type=Path, default=Path("train_settings_NAME.yaml"),
        help="Path to the YAML template file (default: train_settings_NAME.yaml)",
    )
    parser.add_argument(
        "--sh-template", type=Path, default=Path("NAME.sh"),
        help="Path to the sbatch script template file (default: NAME.sh)",
    )
    parser.add_argument(
        "--ranges", type=Path, default=Path("mass_range.txt"),
        help="Path to the mass-range list file (default: mass_range.txt)",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("."),
        help="Directory to write the generated files into (default: current dir)",
    )
    args = parser.parse_args()

    written = generate_files(
        args.yaml_template, args.sh_template, args.ranges, args.outdir
    )

    print(f"Generated {len(written)} files in {args.outdir.resolve()}:")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
