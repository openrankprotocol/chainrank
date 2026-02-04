#!/usr/bin/env python3
"""
Traversed Graph Rankings Analyzer

This script:
1. Loads the traversed graph (seed peers -> neighbours)
2. Loads the scores/rankings
3. Lists rankings of seed peers and their neighbours
"""

import csv
import json
from pathlib import Path

# Zero address to exclude
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def load_traversed_graph(graph_file: Path) -> dict[str, list[str]]:
    """Load traversed graph and return seed -> [neighbours] mapping."""
    print(f"Loading traversed graph from {graph_file}...")

    seed_to_neighbours = {}

    with open(graph_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = row.get("i", "").lower().strip()
            neighbour = row.get("j", "").lower().strip()

            # Skip zero address
            if neighbour == ZERO_ADDRESS:
                continue

            if seed and neighbour:
                if seed not in seed_to_neighbours:
                    seed_to_neighbours[seed] = []
                seed_to_neighbours[seed].append(neighbour)

    print(f"  Loaded {len(seed_to_neighbours)} seed peers")
    total_neighbours = sum(len(n) for n in seed_to_neighbours.values())
    print(f"  Total neighbour connections: {total_neighbours}")

    return seed_to_neighbours


def load_scores(scores_file: Path) -> dict[str, int]:
    """Load scores and return address -> rank mapping."""
    print(f"\nLoading scores from {scores_file}...")

    # Load all addresses with their scores
    addresses_scores = []

    with open(scores_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row.get("i", "").lower().strip()
            score = float(row.get("v", 0))
            if addr:
                addresses_scores.append((addr, score))

    # Scores are already sorted by value descending, so rank = position + 1
    address_to_rank = {}
    for rank, (addr, _) in enumerate(addresses_scores, start=1):
        address_to_rank[addr] = rank

    print(f"  Loaded {len(address_to_rank)} addresses with rankings")

    return address_to_rank


def build_rankings(
    seed_to_neighbours: dict[str, list[str]],
    address_to_rank: dict[str, int],
) -> dict:
    """Build the rankings structure for seed peers and neighbours."""
    print("\nBuilding rankings...")

    rankings = {}

    for seed, neighbours in seed_to_neighbours.items():
        seed_rank = address_to_rank.get(seed)

        neighbour_rankings = []
        for neighbour in neighbours:
            neighbour_rank = address_to_rank.get(neighbour)
            neighbour_rankings.append(
                {
                    "address": neighbour,
                    "rank": neighbour_rank,
                }
            )

        # Sort neighbours by rank (None values at the end)
        neighbour_rankings.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0))

        rankings[seed] = {
            "rank": seed_rank,
            "neighbours": neighbour_rankings,
        }

    # Sort seeds by rank (None values at the end)
    sorted_rankings = dict(
        sorted(
            rankings.items(), key=lambda x: (x[1]["rank"] is None, x[1]["rank"] or 0)
        )
    )

    return sorted_rankings


def print_rankings(rankings: dict):
    """Print rankings in a readable format."""
    print("\n" + "=" * 60)
    print("Seed Peer Rankings")
    print("=" * 60)

    for seed, data in rankings.items():
        seed_rank = data["rank"] if data["rank"] is not None else "unranked"
        print(f"\n{seed}:")
        print(f"  rank: {seed_rank}")
        print(f"  neighbours ({len(data['neighbours'])}):")

        for neighbour in data["neighbours"][:10]:  # Show top 10 neighbours
            n_rank = neighbour["rank"] if neighbour["rank"] is not None else "unranked"
            print(f"    - {neighbour['address']}: rank {n_rank}")

        if len(data["neighbours"]) > 10:
            print(f"    ... and {len(data['neighbours']) - 10} more")


def save_rankings(rankings: dict, output_file: Path):
    """Save rankings to JSON file."""
    print(f"\nSaving rankings to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"  Saved rankings for {len(rankings)} seed peers")


def main():
    # File paths
    graph_file = Path("output/traversed_graph.csv")
    scores_file = Path("scores/ethereum_new_3.csv")
    output_file = Path("output/traversed_graph_rankings.json")

    # Check files exist
    if not graph_file.exists():
        print(f"Error: Graph file not found: {graph_file}")
        return 1

    if not scores_file.exists():
        print(f"Error: Scores file not found: {scores_file}")
        return 1

    print("=" * 60)
    print("Traversed Graph Rankings Analyzer")
    print("=" * 60)

    # Load data
    seed_to_neighbours = load_traversed_graph(graph_file)
    address_to_rank = load_scores(scores_file)

    # Build rankings
    rankings = build_rankings(seed_to_neighbours, address_to_rank)

    # Print rankings
    print_rankings(rankings)

    # Save rankings
    save_rankings(rankings, output_file)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    ranked_seeds = sum(1 for r in rankings.values() if r["rank"] is not None)
    unranked_seeds = len(rankings) - ranked_seeds

    all_neighbours = []
    for data in rankings.values():
        all_neighbours.extend(data["neighbours"])

    ranked_neighbours = sum(1 for n in all_neighbours if n["rank"] is not None)
    unranked_neighbours = len(all_neighbours) - ranked_neighbours

    print(
        f"  Seed peers: {len(rankings)} ({ranked_seeds} ranked, {unranked_seeds} unranked)"
    )
    print(
        f"  Neighbour connections: {len(all_neighbours)} ({ranked_neighbours} ranked, {unranked_neighbours} unranked)"
    )
    print(f"\n✓ Output saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
