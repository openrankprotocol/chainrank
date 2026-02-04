#!/usr/bin/env python3
"""
ChainRank Core Trust Graph Generator
═══════════════════════════════════════════════════════════════════════════════
Filters trust edges to include only addresses in the top 10% of scores.

This creates a "core" trust graph containing only high-trust addresses,
which can be useful for analysis and visualization.

Usage:
    python core_trust_graph.py --chain base
    python core_trust_graph.py --chain ethereum --percentile 5
"""

import argparse
import csv
from pathlib import Path


def load_scores(scores_path: Path) -> list[tuple[str, float]]:
    """Load scores from CSV file and return sorted list of (address, score) tuples."""
    scores = []
    with open(scores_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row["i"].lower()
            score = float(row["v"])
            scores.append((address, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def get_top_percentile(scores: list[tuple[str, float]], percentile: int) -> set[str]:
    """Get addresses in the top percentile of scores."""
    if not scores:
        return set()

    top_count = max(1, len(scores) * percentile // 100)
    top_addresses = {addr for addr, _ in scores[:top_count]}
    return top_addresses


def filter_trust_edges(trust_path: Path, top_addresses: set[str]) -> list[dict]:
    """Filter trust edges to only include addresses in top_addresses."""
    filtered_edges = []
    total_edges = 0

    with open(trust_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_edges += 1
            source = row["i"].lower()
            target = row["j"].lower()

            # Keep edge only if both source and target are in top addresses
            if source in top_addresses and target in top_addresses:
                filtered_edges.append({"i": source, "j": target, "v": row["v"]})

    return filtered_edges, total_edges


def save_filtered_trust(edges: list[dict], output_path: Path):
    """Save filtered trust edges to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "j", "v"])
        writer.writeheader()
        writer.writerows(edges)


def main():
    parser = argparse.ArgumentParser(description="ChainRank Core Trust Graph Generator")
    parser.add_argument(
        "--chain",
        type=str,
        required=True,
        help="Chain to process (e.g., base, ethereum)",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=10,
        help="Top percentile to include (default: 10 for top 10%%)",
    )
    parser.add_argument(
        "--trust-dir",
        type=str,
        default="trust",
        help="Directory containing trust files (default: trust)",
    )
    parser.add_argument(
        "--scores-dir",
        type=str,
        default="scores",
        help="Directory containing scores files (default: scores)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (default: output)",
    )

    args = parser.parse_args()

    print("═" * 79)
    print("ChainRank Core Trust Graph Generator")
    print("═" * 79)

    # Set up paths
    trust_path = Path(args.trust_dir) / f"{args.chain}.csv"
    scores_path = Path(args.scores_dir) / f"{args.chain}.csv"
    output_path = Path(args.output_dir) / f"{args.chain}_core_trust.csv"

    # Validate input files exist
    if not trust_path.exists():
        print(f"Error: Trust file not found: {trust_path}")
        return 1

    if not scores_path.exists():
        print(f"Error: Scores file not found: {scores_path}")
        return 1

    print(f"\n{'─' * 79}")
    print("Configuration")
    print(f"{'─' * 79}")
    print(f"  Chain: {args.chain}")
    print(f"  Top percentile: {args.percentile}%")
    print(f"  Trust file: {trust_path}")
    print(f"  Scores file: {scores_path}")
    print(f"  Output file: {output_path}")

    # Load scores
    print(f"\n{'─' * 79}")
    print("Loading Scores")
    print(f"{'─' * 79}")
    scores = load_scores(scores_path)
    print(f"  Total addresses with scores: {len(scores):,}")

    # Get top percentile addresses
    print(f"\n{'─' * 79}")
    print(f"Filtering Top {args.percentile}% Addresses")
    print(f"{'─' * 79}")
    top_addresses = get_top_percentile(scores, args.percentile)
    print(f"  Addresses in top {args.percentile}%: {len(top_addresses):,}")

    # Show some top addresses
    if scores:
        print(f"\n  Top 5 addresses by score:")
        for i, (addr, score) in enumerate(scores[:5], 1):
            print(f"    {i}. {addr}: {score:.6f}")

    # Filter trust edges
    print(f"\n{'─' * 79}")
    print("Filtering Trust Edges")
    print(f"{'─' * 79}")
    filtered_edges, total_edges = filter_trust_edges(trust_path, top_addresses)
    print(f"  Total trust edges: {total_edges:,}")
    print(
        f"  Filtered edges (both addresses in top {args.percentile}%): {len(filtered_edges):,}"
    )

    if total_edges > 0:
        retention_rate = len(filtered_edges) / total_edges * 100
        print(f"  Retention rate: {retention_rate:.2f}%")

    # Save filtered trust graph
    print(f"\n{'─' * 79}")
    print("Saving Core Trust Graph")
    print(f"{'─' * 79}")
    save_filtered_trust(filtered_edges, output_path)
    print(f"  ✓ Saved to {output_path}")

    # Summary statistics
    if filtered_edges:
        unique_sources = len(set(e["i"] for e in filtered_edges))
        unique_targets = len(set(e["j"] for e in filtered_edges))
        unique_addresses = len(
            set(e["i"] for e in filtered_edges) | set(e["j"] for e in filtered_edges)
        )

        print(f"\n{'─' * 79}")
        print("Summary")
        print(f"{'─' * 79}")
        print(f"  Unique source addresses: {unique_sources:,}")
        print(f"  Unique target addresses: {unique_targets:,}")
        print(f"  Total unique addresses: {unique_addresses:,}")
        print(
            f"  Average edges per address: {len(filtered_edges) / unique_addresses:.2f}"
        )

    print(f"\n{'═' * 79}")
    print("Done!")
    print("═" * 79)

    return 0


if __name__ == "__main__":
    exit(main())
