#!/usr/bin/env python3
"""
Traverse Local Trust

This script:
1. Loads top accounts from analysed_scores_{chain}.csv (pre-filtered EOAs with identity)
2. Gets outgoing connections from seed peers (neighbours)
3. Filters out neighbours that are smart contracts
4. Saves results to output/traversed_graph.csv
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import toml
from dotenv import load_dotenv

load_dotenv()

DUNE_API_KEY = os.getenv("DUNE_API_KEY")
DUNE_ADDRESSES_QUERY_ID = 6467362

# Load config for parallel settings
CONFIG_PATH = Path("config.toml")
if CONFIG_PATH.exists():
    CONFIG = toml.load(CONFIG_PATH)
    PARALLEL_REQUESTS = CONFIG.get("indexer", {}).get("parallel_requests", 50)
else:
    PARALLEL_REQUESTS = 50


def get_rpc_url(chain: str) -> str:
    """Get RPC URL for the specified chain."""
    if chain == "ethereum":
        return os.getenv("RPC_ETHEREUM", "https://ethereum.publicnode.com")
    elif chain == "base":
        return os.getenv("RPC_BASE", "https://base.publicnode.com")
    else:
        return os.getenv(f"RPC_{chain.upper()}", f"https://{chain}.publicnode.com")


def is_eoa_single(address: str, rpc_url: str) -> tuple[str, bool]:
    """Check if a single address is an EOA. Returns (address, is_eoa)."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getCode",
        "params": [address, "latest"],
        "id": 1,
    }

    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        result = response.json().get("result", "0x")
        return (address, result == "0x")
    except Exception:
        return (address, True)  # Assume EOA if check fails


def is_eoa_batch(
    addresses: list[str], rpc_url: str, show_progress: bool = False
) -> dict[str, bool]:
    """Check multiple addresses in parallel. Returns dict of address -> is_eoa."""
    results = {}
    total = len(addresses)
    completed = 0
    contracts = 0

    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
        futures = {
            executor.submit(is_eoa_single, addr, rpc_url): addr for addr in addresses
        }

        for future in as_completed(futures):
            addr, is_eoa = future.result()
            results[addr] = is_eoa
            completed += 1
            if not is_eoa:
                contracts += 1

            if show_progress and completed % 500 == 0:
                print(
                    f"    Progress: {completed}/{total} checked - EOAs: {completed - contracts}, Contracts: {contracts}"
                )

    return results


def load_analysed_scores(analysed_path: Path) -> list[tuple[str, float]]:
    """Load addresses from analysed_scores CSV file (already filtered EOAs with identity)."""
    scores = []
    with open(analysed_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("address", "").lower().strip()
            trust_score = row.get("trust_score", "0")
            try:
                score = float(trust_score)
            except ValueError:
                score = 0.0
            if address and address.startswith("0x") and len(address) == 42:
                scores.append((address, score))

    # Already sorted by rank in the file, but sort again to be safe
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def load_seed_peers_from_analysed(
    analysed_path: Path, target_count: int = 100
) -> set[str]:
    """
    Load seed peer addresses from analysed_scores CSV file.
    These are already pre-filtered EOAs with identity verification.
    """
    print(f"Loading seed peers from {analysed_path}...")
    scores = load_analysed_scores(analysed_path)
    print(f"  Loaded {len(scores)} pre-filtered addresses")

    # Take top target_count addresses
    seed_peers = set()
    for addr, score in scores[:target_count]:
        seed_peers.add(addr)

    print(f"  Selected {len(seed_peers)} seed peers")
    return seed_peers


def load_seed_peers_from_file(file_path: Path, limit: int = 1000) -> set[str]:
    """Load seed peer addresses from local CSV file."""
    print(f"Loading seed peers from {file_path}...")

    addresses = set()
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row.get("address") or row.get("i") or ""
            addr = addr.lower().strip()
            if addr and addr.startswith("0x") and len(addr) == 42:
                addresses.add(addr)
                if len(addresses) >= limit:
                    break

    print(f"  Loaded {len(addresses)} seed peers")
    return addresses


def filter_eoas(addresses: set[str], rpc_url: str) -> set[str]:
    """Filter addresses to keep only EOAs (exclude smart contracts) using parallel calls."""
    print(
        f"\nFiltering EOAs from {len(addresses)} addresses (parallel: {PARALLEL_REQUESTS} workers)..."
    )

    addr_list = list(addresses)
    results = is_eoa_batch(addr_list, rpc_url, show_progress=True)

    eoas = {addr for addr, is_eoa in results.items() if is_eoa}
    contracts = len(addresses) - len(eoas)

    print(f"  Final: {len(eoas)} EOAs, {contracts} contracts excluded")
    return eoas


def filter_high_degree_seeds(
    seed_peers: set[str],
    outgoing_index: dict[str, list[tuple[str, float]]],
    max_arcs: int = 400,
) -> set[str]:
    """Filter out seed peers with more than max_arcs outgoing arcs."""
    print(f"\nFiltering seed peers with >{max_arcs} outgoing arcs...")

    filtered = set()
    excluded = 0

    for addr in seed_peers:
        outgoing_count = len(outgoing_index.get(addr, []))

        if outgoing_count > max_arcs:
            excluded += 1
            print(f"  Excluding {addr}: {outgoing_count} outgoing arcs")
        else:
            filtered.add(addr)

    print(f"  Final: {len(filtered)} kept, {excluded} excluded")
    return filtered


def load_local_trust(trust_file: Path) -> list[tuple[str, str, float]]:
    """Load local trust arcs from CSV file."""
    print(f"\nLoading local trust from {trust_file}...")

    arcs = []
    with open(trust_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = row.get("i", "").lower().strip()
            j = row.get("j", "").lower().strip()
            v = float(row.get("v", 0))
            if i and j:
                arcs.append((i, j, v))

    print(f"  Loaded {len(arcs)} trust arcs")
    return arcs


def build_outgoing_index(
    arcs: list[tuple[str, str, float]],
) -> dict[str, list[tuple[str, float]]]:
    """Build index of outgoing arcs: address -> [(target, value), ...]"""
    index = {}
    for i, j, v in arcs:
        if i not in index:
            index[i] = []
        index[i].append((j, v))
    return index


def build_incoming_index(
    arcs: list[tuple[str, str, float]],
) -> dict[str, list[tuple[str, float]]]:
    """Build index of incoming arcs: address -> [(source, value), ...]"""
    index = {}
    for i, j, v in arcs:
        if j not in index:
            index[j] = []
        index[j].append((i, v))
    return index


# Zero address to exclude
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def get_outgoing_arcs(
    seed_peers: set[str],
    outgoing_index: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, str, float]]:
    """Get all outgoing arcs from seed peers."""
    print("\nGetting outgoing arcs from seed peers...")

    outgoing_arcs = []
    for peer in seed_peers:
        if peer in outgoing_index:
            for target, value in outgoing_index[peer]:
                if target != ZERO_ADDRESS:
                    outgoing_arcs.append((peer, target, value))

    print(f"  Found {len(outgoing_arcs)} outgoing arcs")
    return outgoing_arcs


def get_incoming_arcs(
    seed_peers: set[str],
    incoming_index: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, str, float]]:
    """Get all incoming arcs to seed peers."""
    print("\nGetting incoming arcs to seed peers...")

    incoming_arcs = []
    for peer in seed_peers:
        if peer in incoming_index:
            for source, value in incoming_index[peer]:
                if source != ZERO_ADDRESS:
                    incoming_arcs.append((source, peer, value))

    print(f"  Found {len(incoming_arcs)} incoming arcs")
    return incoming_arcs


def filter_arcs_by_eoa_neighbours(
    arcs: list[tuple[str, str, float]],
    seed_peers: set[str],
    rpc_url: str,
) -> list[tuple[str, str, float]]:
    """Filter arcs to only keep those where the target (neighbour) is an EOA."""
    print("\nFiltering neighbours (targets) for contracts...")

    # Get all unique neighbours (targets that are not seed peers)
    neighbours = set()
    for i, j, v in arcs:
        if j not in seed_peers:
            neighbours.add(j)

    print(f"  Found {len(neighbours)} unique neighbours")

    # Check which neighbours are EOAs
    print(
        f"  Checking {len(neighbours)} neighbours for contracts (parallel: {PARALLEL_REQUESTS} workers)..."
    )
    eoa_results = is_eoa_batch(list(neighbours), rpc_url, show_progress=True)

    eoa_neighbours = {addr for addr, is_eoa in eoa_results.items() if is_eoa}
    contracts = len(neighbours) - len(eoa_neighbours)

    print(f"  {len(eoa_neighbours)} EOA neighbours, {contracts} contracts excluded")

    # Valid targets = EOA neighbours + seed peers (already filtered)
    valid_targets = eoa_neighbours | seed_peers

    # Filter arcs
    filtered_arcs = [(i, j, v) for i, j, v in arcs if j in valid_targets]
    print(f"  Arcs after filtering: {len(filtered_arcs)} (from {len(arcs)})")

    return filtered_arcs


def filter_arcs_by_eoa_sources(
    arcs: list[tuple[str, str, float]],
    seed_peers: set[str],
    rpc_url: str,
) -> list[tuple[str, str, float]]:
    """Filter arcs to only keep those where the source is an EOA (for incoming arcs)."""
    print("\nFiltering sources for contracts...")

    # Get all unique sources (that are not seed peers)
    sources = set()
    for i, j, v in arcs:
        if i not in seed_peers:
            sources.add(i)

    print(f"  Found {len(sources)} unique sources")

    # Check which sources are EOAs
    print(
        f"  Checking {len(sources)} sources for contracts (parallel: {PARALLEL_REQUESTS} workers)..."
    )
    eoa_results = is_eoa_batch(list(sources), rpc_url, show_progress=True)

    eoa_sources = {addr for addr, is_eoa in eoa_results.items() if is_eoa}
    contracts = len(sources) - len(eoa_sources)

    print(f"  {len(eoa_sources)} EOA sources, {contracts} contracts excluded")

    # Valid sources = EOA sources + seed peers (already filtered)
    valid_sources = eoa_sources | seed_peers

    # Filter arcs
    filtered_arcs = [(i, j, v) for i, j, v in arcs if i in valid_sources]
    print(f"  Arcs after filtering: {len(filtered_arcs)} (from {len(arcs)})")

    return filtered_arcs


def save_arcs(arcs: list[tuple[str, str, float]], output_file: Path):
    """Save trust arcs to CSV file."""
    print(f"\nSaving arcs to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "v"])
        for i, j, v in arcs:
            writer.writerow([i, j, v])

    print(f"  Saved {len(arcs)} arcs")


def main():
    parser = argparse.ArgumentParser(description="Traverse local trust from seed peers")
    parser.add_argument(
        "--chain",
        type=str,
        default="ethereum",
        help="Chain to process (e.g., ethereum, base)",
    )
    parser.add_argument(
        "--trust-file",
        type=str,
        default=None,
        help="Path to local trust CSV file (default: trust/{chain}.csv)",
    )
    parser.add_argument(
        "--analysed-file",
        type=str,
        default=None,
        help="Path to analysed scores CSV file (default: analysis/output/analysed_scores_{chain}.csv)",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        default=None,
        help="Path to local seed peers CSV file (overrides --scores-file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for filtered arcs (default: output/{chain}_traversed_graph.csv)",
    )
    parser.add_argument(
        "--target-eoas",
        type=int,
        default=100,
        help="Target number of EOA seed peers to collect (default: 100)",
    )

    args = parser.parse_args()

    # Set default paths based on chain
    trust_file = (
        Path(args.trust_file) if args.trust_file else Path(f"trust/{args.chain}.csv")
    )
    analysed_file = (
        Path(args.analysed_file)
        if args.analysed_file
        else Path(f"analysis/output/analysed_scores_{args.chain}.csv")
    )
    output_file = (
        Path(args.output)
        if args.output
        else Path(f"analysis/output/{args.chain}_traversed_graph.csv")
    )
    seed_file = Path(args.seed_file) if args.seed_file else None

    rpc_url = get_rpc_url(args.chain)

    if not trust_file.exists():
        print(f"Error: Trust file not found: {trust_file}")
        return 1

    print("=" * 60)
    print("Traverse Local Trust")
    print("=" * 60)
    print(f"  Chain: {args.chain}")
    print(f"  RPC URL: {rpc_url}")
    print(f"  Trust file: {trust_file}")
    print(f"  Target EOAs: {args.target_eoas}")

    # Load seed peers - either from seed file or from analysed scores
    if seed_file and seed_file.exists():
        print(f"  Seed file: {seed_file}")
        seed_peers = load_seed_peers_from_file(seed_file, limit=args.target_eoas * 2)
        # Filter out smart contracts from seed peers
        seed_peers = filter_eoas(seed_peers, rpc_url)
        # Limit to target count
        seed_peers = set(list(seed_peers)[: args.target_eoas])
    else:
        if not analysed_file.exists():
            print(f"Error: Analysed scores file not found: {analysed_file}")
            print(
                f"  Run 'python analysis/address_stats.py --chain {args.chain}' first"
            )
            return 1
        print(f"  Analysed file: {analysed_file}")
        # Load seed peers from pre-filtered analysed scores (already EOAs with identity)
        seed_peers = load_seed_peers_from_analysed(
            analysed_file, target_count=args.target_eoas
        )

    # Load local trust (need it first to filter high-degree seeds)
    arcs = load_local_trust(trust_file)

    # Build outgoing and incoming indexes
    print("\nBuilding arc indexes...")
    outgoing_index = build_outgoing_index(arcs)
    incoming_index = build_incoming_index(arcs)
    print(f"  Indexed {len(outgoing_index)} source addresses (outgoing)")
    print(f"  Indexed {len(incoming_index)} target addresses (incoming)")

    # Filter out high-degree seed peers (>400 outgoing arcs)
    seed_peers = filter_high_degree_seeds(seed_peers, outgoing_index, max_arcs=400)

    # Print seed peers
    print(f"\nFiltered seed peers ({len(seed_peers)}):")
    for i, peer in enumerate(sorted(seed_peers), 1):
        print(f"  {i}. {peer}")

    # Save seed peers
    seed_peers_file = Path(f"analysis/output/{args.chain}_filtered_seed_peers.csv")
    seed_peers_file.parent.mkdir(parents=True, exist_ok=True)
    with open(seed_peers_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["address"])
        for peer in sorted(seed_peers):
            writer.writerow([peer])
    print(f"\n  Saved {len(seed_peers)} seed peers to {seed_peers_file}")

    # Get outgoing arcs from seed peers
    outgoing_arcs = get_outgoing_arcs(seed_peers, outgoing_index)

    # Get incoming arcs to seed peers
    incoming_arcs = get_incoming_arcs(seed_peers, incoming_index)

    # Combine and deduplicate arcs (no smart contract filtering)
    all_arcs_set = set()
    for arc in outgoing_arcs + incoming_arcs:
        all_arcs_set.add(arc)
    filtered_arcs = list(all_arcs_set)
    print(
        f"\nCombined arcs: {len(filtered_arcs)} (outgoing: {len(outgoing_arcs)}, incoming: {len(incoming_arcs)})"
    )

    # Save results
    save_arcs(filtered_arcs, output_file)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Seed peers: {len(seed_peers)}")
    print(f"  Outgoing arcs: {len(outgoing_arcs)}")
    print(f"  Incoming arcs: {len(incoming_arcs)}")
    print(f"  Total combined arcs: {len(filtered_arcs)}")
    print(f"\n✓ Output saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
