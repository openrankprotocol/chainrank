#!/usr/bin/env python3
"""
ChainRank Score Processor
═══════════════════════════════════════════════════════════════════════════════
Maps addresses to ENS names in score files (i,v format) and saves to output folder.

Usage:
    python process_scores.py
    python process_scores.py --input scores/ethereum.csv
    python process_scores.py --ens raw/ens_names.csv
"""

import argparse
import csv
from pathlib import Path

import toml
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "r") as f:
        return toml.load(f)


def load_ens_mapping(ens_path: Path) -> dict[str, str]:
    """Load ENS names mapping from CSV file."""
    mapping = {}
    with open(ens_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("address", "").lower()
            ens_name = row.get("ens_name", "")
            if address and ens_name:
                mapping[address] = ens_name
    return mapping


def find_ens_file(raw_folder: Path) -> Path | None:
    """Find the ENS names CSV file."""
    ens_path = raw_folder / "ens_names.csv"
    if ens_path.exists():
        return ens_path
    return None


def process_score_file(
    input_path: Path, output_path: Path, ens_mapping: dict[str, str]
):
    """Process a score file and map addresses to ENS names."""
    rows = []

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i_addr = row.get("i", "").lower()
            v = row.get("v", "")

            # Map address to ENS name if available
            i_name = ens_mapping.get(i_addr, i_addr)

            rows.append({"i": i_name, "v": v})

    # Save to output
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "v"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="ChainRank Score Processor")
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config.toml file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input score CSV file (default: all files in scores/)",
    )
    parser.add_argument(
        "--ens",
        type=str,
        default=None,
        help="Path to ENS names CSV file (default: raw/ens_names.csv)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(config_path)

    print("═" * 79)
    print("ChainRank Score Processor")
    print("═" * 79)

    # Setup folders
    raw_folder = Path(config.get("output", {}).get("raw_folder", "raw"))
    scores_folder = Path("scores")
    output_folder = Path("output")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load ENS mapping
    if args.ens:
        ens_path = Path(args.ens)
    else:
        ens_path = find_ens_file(raw_folder)

    if not ens_path or not ens_path.exists():
        print("Error: ENS file not found")
        return 1

    print(f"ENS file: {ens_path}")
    ens_mapping = load_ens_mapping(ens_path)
    print(f"Loaded {len(ens_mapping):,} ENS mappings")
    print()

    # Find input files
    if args.input:
        input_paths = [Path(args.input)]
    else:
        input_paths = sorted(scores_folder.glob("*.csv"))

    if not input_paths:
        print("Error: No input files found")
        return 1

    print("─" * 79)
    print("Processing Score Files")
    print("─" * 79)

    for input_path in input_paths:
        output_path = output_folder / input_path.name
        row_count = process_score_file(input_path, output_path, ens_mapping)
        print(f"  ✓ {input_path.name} -> {output_path.name} ({row_count:,} rows)")

    print()
    print("═" * 79)
    print("✓ Done!")
    print("═" * 79)

    return 0


if __name__ == "__main__":
    exit(main())
