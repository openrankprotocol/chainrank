#!/usr/bin/env python3
"""
ChainRank Score Processor
═══════════════════════════════════════════════════════════════════════════════
Maps addresses to ENS names, Farcaster names, and CEX names in score files (i,v format) and saves to output folder.

Usage:
    python process_scores.py
    python process_scores.py --input scores/ethereum.csv
    python process_scores.py --ens raw/ens_names_2025.csv
"""

import argparse
import csv
import json
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


def find_ens_files(raw_folder: Path) -> list[Path]:
    """Find all ENS names CSV files (ens_names_*.csv)."""
    return sorted(raw_folder.glob("ens_names_*.csv"))


def load_cex_addresses(raw_folder: Path) -> set[str]:
    """Load CEX addresses from cex_addresses.csv."""
    cex_path = raw_folder / "cex_addresses.csv"
    cex_addresses = set()
    if not cex_path.exists():
        return cex_addresses
    with open(cex_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("address", "").lower()
            if address:
                cex_addresses.add(address)
    return cex_addresses


def load_ens_mapping_from_files(ens_paths: list[Path]) -> dict[str, str]:
    """Load ENS names mapping from multiple CSV files, merging them."""
    mapping = {}
    for ens_path in ens_paths:
        with open(ens_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                address = row.get("address", "").lower()
                ens_name = row.get("ens_name", "")
                if address and ens_name:
                    mapping[address] = ens_name
    return mapping


def load_contracts(raw_folder: Path) -> set[str]:
    """Load contract addresses from eth_contracts.csv."""
    contracts_path = raw_folder / "eth_contracts.csv"
    contracts = set()
    if not contracts_path.exists():
        return contracts
    with open(contracts_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("address", "").lower()
            if address:
                contracts.add(address)
    return contracts


def load_wrld_mapping(raw_folder: Path) -> dict[str, str]:
    """Load World ID addresses mapping from wrld_addresses.csv."""
    wrld_path = raw_folder / "wrld_addresses.csv"
    mapping = {}
    if not wrld_path.exists():
        return mapping
    with open(wrld_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("address", "").lower()
            if address:
                mapping[address] = f"{address}.wrld"
    return mapping


def load_fc_mapping(raw_folder: Path) -> dict[str, str]:
    """Load Farcaster addresses mapping from fc_addresses.csv."""
    fc_path = raw_folder / "fc_addresses.csv"
    mapping = {}
    if not fc_path.exists():
        return mapping
    with open(fc_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("fname", "")
            addresses_json = row.get("verified_addresses", "[]")
            if not fname:
                continue
            try:
                addresses = json.loads(addresses_json)
                for addr in addresses:
                    # Only include 20-byte Ethereum addresses (42 chars with 0x)
                    if addr and len(addr) == 42 and addr.startswith("0x"):
                        addr_lower = addr.lower()
                        # Don't overwrite if already mapped
                        if addr_lower not in mapping:
                            mapping[addr_lower] = f"{fname}.fc"
            except json.JSONDecodeError:
                pass
    return mapping


def process_score_file(
    input_path: Path,
    output_path: Path,
    ens_mapping: dict[str, str],
    cex_addresses: set[str],
    fc_mapping: dict[str, str],
    wrld_mapping: dict[str, str],
    contracts: set[str],
):
    """Process a score file and map addresses to ENS/FC/WRLD names, filtering out contracts and CEX addresses."""
    rows = []
    filtered_count = 0
    cex_filtered_count = 0

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i_addr = row.get("i", "").lower()
            v = row.get("v", "")

            # Skip contract addresses
            if i_addr in contracts:
                filtered_count += 1
                continue

            # Skip CEX addresses
            if i_addr in cex_addresses:
                cex_filtered_count += 1
                continue

            # Map address: ENS first, then FC, then WRLD, fallback to address
            if i_addr in ens_mapping:
                i_name = ens_mapping[i_addr]
            elif i_addr in fc_mapping:
                i_name = fc_mapping[i_addr]
            elif i_addr in wrld_mapping:
                i_name = wrld_mapping[i_addr]
            else:
                i_name = i_addr

            rows.append({"i": i_name, "v": v})

    # Save to output
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "v"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), filtered_count, cex_filtered_count


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
        help="Path to specific ENS names CSV file (default: all ens_names_*.csv in raw folder)",
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
        ens_paths = [Path(args.ens)]
    else:
        ens_paths = find_ens_files(raw_folder)

    if not ens_paths:
        print("Error: No ENS files found")
        return 1

    print(f"ENS files: {len(ens_paths)} file(s)")
    for p in ens_paths:
        print(f"  - {p.name}")
    ens_mapping = load_ens_mapping_from_files(ens_paths)
    print(f"Loaded {len(ens_mapping):,} ENS mappings")

    # Load CEX addresses to filter
    cex_addresses = load_cex_addresses(raw_folder)
    print(f"Loaded {len(cex_addresses):,} CEX addresses to filter")

    # Load Farcaster mapping
    fc_mapping = load_fc_mapping(raw_folder)
    print(f"Loaded {len(fc_mapping):,} Farcaster mappings")

    # Load World ID mapping
    wrld_mapping = load_wrld_mapping(raw_folder)
    print(f"Loaded {len(wrld_mapping):,} World ID mappings")

    # Load contract addresses to filter
    contracts = load_contracts(raw_folder)
    print(f"Loaded {len(contracts):,} contract addresses to filter")
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
        row_count, filtered_count, cex_filtered_count = process_score_file(
            input_path,
            output_path,
            ens_mapping,
            cex_addresses,
            fc_mapping,
            wrld_mapping,
            contracts,
        )
        print(
            f"  ✓ {input_path.name} -> {output_path.name} ({row_count:,} rows, {filtered_count:,} contracts filtered, {cex_filtered_count:,} CEX filtered)"
        )

    print()
    print("═" * 79)
    print("✓ Done!")
    print("═" * 79)

    return 0


if __name__ == "__main__":
    exit(main())
