#!/usr/bin/env python3
"""
Map addresses from scores files to their ENS/Farcaster identities.

Priority: ENS > Farcaster

Usage:
    python map_identities.py --scores scores/ethereum.csv --output scores/ethereum_mapped.csv
    python map_identities.py --scores scores/defi.csv --output scores/defi_mapped.csv
"""

import argparse
import ast
import csv
from pathlib import Path


def load_ens_names(raw_folder: Path) -> dict[str, str]:
    """Load ENS names from all ENS files in raw folder.

    Returns a dict mapping lowercase address -> ENS name.
    """
    ens_map = {}

    # Load from ens_names_*.csv files
    for ens_file in raw_folder.glob("ens_names_*.csv"):
        print(f"  Loading {ens_file.name}...")
        with open(ens_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                address = row.get("address", "").lower().strip()
                ens_name = row.get("ens_name", "").strip()
                if address and ens_name and address.startswith("0x"):
                    # Only store if we don't already have one (first wins)
                    if address not in ens_map:
                        ens_map[address] = ens_name

    # Also load from dune_ens_names.csv if it exists
    dune_file = raw_folder / "dune_ens_names.csv"
    if dune_file.exists():
        print(f"  Loading {dune_file.name}...")
        with open(dune_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                address = row.get("address", "").lower().strip()
                ens_name = row.get("name", "").strip()
                if address and ens_name and address.startswith("0x"):
                    if address not in ens_map:
                        ens_map[address] = ens_name

    return ens_map


def load_farcaster_names(raw_folder: Path) -> dict[str, str]:
    """Load Farcaster names from fc_addresses.csv.

    Returns a dict mapping lowercase address -> Farcaster username.
    """
    fc_map = {}

    fc_file = raw_folder / "fc_addresses.csv"
    if not fc_file.exists():
        print(f"  Warning: {fc_file} not found")
        return fc_map

    print(f"  Loading {fc_file.name}...")
    with open(fc_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("fname", "").strip()
            verified_addresses_str = row.get("verified_addresses", "[]")

            if not fname:
                continue

            # Parse the JSON-like array of addresses
            try:
                # The format is like: ["0x...", "0x..."]
                addresses = ast.literal_eval(verified_addresses_str)
                if isinstance(addresses, list):
                    for addr in addresses:
                        addr = str(addr).lower().strip()
                        # Only include valid Ethereum addresses (42 chars: 0x + 40 hex)
                        if addr.startswith("0x") and len(addr) == 42:
                            if addr not in fc_map:
                                fc_map[addr] = fname
            except (ValueError, SyntaxError):
                # Skip malformed entries
                continue

    return fc_map


def load_scores(scores_path: Path) -> list[dict]:
    """Load scores from CSV file.

    Returns list of dicts with 'address' (renamed from 'i') and 'score' (renamed from 'v').
    """
    scores = []

    with open(scores_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both old format (i,v) and potential new format (address,score)
            address = row.get("i") or row.get("address", "")
            score = row.get("v") or row.get("score", "0")

            address = address.lower().strip()
            if address:
                scores.append({"address": address, "score": score})

    return scores


def map_identities(
    scores: list[dict], ens_map: dict[str, str], fc_map: dict[str, str]
) -> list[dict]:
    """Map addresses to their identities.

    Priority: ENS > Farcaster

    Returns list of dicts with address, score, identity, and identity_type.
    """
    mapped = []

    for entry in scores:
        address = entry["address"]
        score = entry["score"]

        # Check ENS first (higher priority)
        if address in ens_map:
            identity = ens_map[address]
            identity_type = "ens"
        # Then check Farcaster
        elif address in fc_map:
            identity = fc_map[address]
            identity_type = "farcaster"
        else:
            identity = ""
            identity_type = ""

        mapped.append(
            {
                "address": address,
                "score": score,
                "identity": identity,
                "identity_type": identity_type,
            }
        )

    return mapped


def save_mapped_scores(mapped: list[dict], output_path: Path):
    """Save mapped scores to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["address", "score", "identity", "identity_type"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapped)


def main():
    parser = argparse.ArgumentParser(
        description="Map addresses from scores to ENS/Farcaster identities"
    )
    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to scores CSV file (e.g., scores/ethereum.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for mapped scores (default: scores/{name}_mapped.csv)",
    )
    parser.add_argument(
        "--raw-folder",
        type=str,
        default="raw",
        help="Path to raw folder containing ENS and Farcaster files (default: raw)",
    )

    args = parser.parse_args()

    scores_path = Path(args.scores)
    raw_folder = Path(args.raw_folder)

    if args.output:
        output_path = Path(args.output)
    else:
        # Default: same folder, append _mapped before extension
        output_path = scores_path.parent / f"{scores_path.stem}_mapped.csv"

    if not scores_path.exists():
        print(f"Error: Scores file not found: {scores_path}")
        return 1

    if not raw_folder.exists():
        print(f"Error: Raw folder not found: {raw_folder}")
        return 1

    print("=" * 60)
    print("Address Identity Mapper")
    print("=" * 60)
    print(f"Scores file: {scores_path}")
    print(f"Raw folder: {raw_folder}")
    print(f"Output: {output_path}")
    print()

    # Load identity mappings
    print("Loading ENS names...")
    ens_map = load_ens_names(raw_folder)
    print(f"  Total: {len(ens_map):,} ENS names")
    print()

    print("Loading Farcaster names...")
    fc_map = load_farcaster_names(raw_folder)
    print(f"  Total: {len(fc_map):,} Farcaster addresses")
    print()

    # Load scores
    print("Loading scores...")
    scores = load_scores(scores_path)
    print(f"  Total: {len(scores):,} addresses")
    print()

    # Map identities
    print("Mapping identities...")
    mapped = map_identities(scores, ens_map, fc_map)

    # Count statistics
    ens_count = sum(1 for m in mapped if m["identity_type"] == "ens")
    fc_count = sum(1 for m in mapped if m["identity_type"] == "farcaster")
    no_identity = sum(1 for m in mapped if not m["identity_type"])

    print(f"  ENS matches: {ens_count:,}")
    print(f"  Farcaster matches: {fc_count:,}")
    print(f"  No identity: {no_identity:,}")
    print()

    # Save output
    print("Saving mapped scores...")
    save_mapped_scores(mapped, output_path)
    print(f"  Saved to: {output_path}")
    print()

    print("=" * 60)
    print("✓ Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
