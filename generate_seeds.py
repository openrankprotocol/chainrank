#!/usr/bin/env python3
"""
ChainRank Seed Score Generator
═══════════════════════════════════════════════════════════════════════════════
Generates equal seed scores for all seed contracts defined in config.toml.
Each seed contract gets an equal share such that all scores sum to 1.0.

Usage:
    python generate_seeds.py --config config.toml
    python generate_seeds.py --config config.toml --chain ethereum
"""

import argparse
import csv
from pathlib import Path

import toml


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "r") as f:
        return toml.load(f)


def generate_seeds(
    config: dict, chain_filter: str | None = None
) -> dict[str, list[tuple[str, float]]]:
    """
    Generate seed scores for all seed contracts.

    Args:
        config: Loaded config dictionary
        chain_filter: Optional chain to filter by (e.g., 'ethereum')

    Returns:
        Dictionary mapping chain -> list of (address, score) tuples
    """
    seed_contracts = config.get("seed_contracts", {})

    # Collect all addresses per chain
    chain_addresses: dict[
        str, list[tuple[str, str]]
    ] = {}  # chain -> [(protocol, address)]

    for protocol_name, chain_addresses_dict in seed_contracts.items():
        if isinstance(chain_addresses_dict, dict):
            for chain, address in chain_addresses_dict.items():
                if chain_filter and chain.lower() != chain_filter.lower():
                    continue
                if chain not in chain_addresses:
                    chain_addresses[chain] = []
                chain_addresses[chain].append((protocol_name, address.lower()))

    # Calculate equal scores per chain
    result: dict[str, list[tuple[str, float, str]]] = {}

    for chain, protocols in chain_addresses.items():
        num_contracts = len(protocols)
        if num_contracts == 0:
            continue

        equal_score = 1.0 / num_contracts
        result[chain] = [(protocol, addr, equal_score) for protocol, addr in protocols]

    return result


def save_seeds(
    seeds: dict[str, list[tuple[str, str, float]]], output_folder: str
) -> None:
    """
    Save seed scores to CSV files.

    Args:
        seeds: Dictionary mapping chain -> list of (protocol, address, score) tuples
        output_folder: Folder to save CSV files
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for chain, seed_list in seeds.items():
        filename = output_path / f"{chain}_seeds.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["i", "v"])

            for protocol, address, score in seed_list:
                writer.writerow([address, f"{score:.10f}"])

        print(f"  ✓ Saved {len(seed_list)} seeds to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="ChainRank Seed Score Generator - Generate equal seed scores for seed contracts"
    )
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config.toml file"
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Filter by specific chain (e.g., ethereum)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seed",
        help="Output folder for seed files (default: seed)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print("═" * 60)
    print("ChainRank Seed Score Generator")
    print("═" * 60)

    config = load_config(config_path)

    # Generate seeds
    print("\nGenerating seed scores...")
    seeds = generate_seeds(config, args.chain)

    if not seeds:
        print("No seed contracts found in config!")
        return 1

    # Print summary
    print("\nSeed Contract Summary:")
    print("-" * 40)
    total_contracts = 0
    for chain, seed_list in seeds.items():
        num = len(seed_list)
        total_contracts += num
        score = seed_list[0][2] if seed_list else 0
        print(f"  {chain}: {num} contracts @ {score:.6f} each")
    print(f"  Total: {total_contracts} contracts")

    # Verify scores sum to 1.0 per chain
    print("\nVerifying scores...")
    for chain, seed_list in seeds.items():
        total = sum(score for _, _, score in seed_list)
        if abs(total - 1.0) > 1e-9:
            print(f"  ⚠ Warning: {chain} scores sum to {total:.10f}, not 1.0")
        else:
            print(f"  ✓ {chain}: scores sum to {total:.10f}")

    # Save seeds
    print(f"\nSaving to {args.output}/...")
    save_seeds(seeds, args.output)

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
