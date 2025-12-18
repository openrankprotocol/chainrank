#!/usr/bin/env python3
"""
ChainRank Local Trust Calculator
═══════════════════════════════════════════════════════════════════════════════
Calculates local trust scores from raw on-chain events using EigenTrust formula.

Formula: trust = max(0, Σ log(amount) * event_weight)

ENS Verification: Addresses with ENS names receive a 2x multiplier on incoming trust.

Usage:
    python calculate_trust.py --config config.toml
    python calculate_trust.py --config config.toml --input raw/ethereum_aave_20240101_120000.csv
    python calculate_trust.py --config config.toml --ens raw/ens_names_20251217_163445.csv
"""

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import toml
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL DECIMALS
# ═══════════════════════════════════════════════════════════════════════════════

PROTOCOL_DECIMALS = {
    "usdc": 6,
    "usdt": 6,
    "wbtc": 8,
    "dai": 18,
    "weth": 18,
    # Default for most tokens (ETH, AAVE, etc.) is 18
}


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "r") as f:
        return toml.load(f)


def find_raw_files_by_chain_year_month(
    raw_folder: Path, chain: str, year: int, month: int
) -> list[Path]:
    """Find all raw event CSV files for a specific chain, year, and month."""
    pattern = f"{chain}_*_{year}_{month:02d}.csv"
    csv_files = list(raw_folder.glob(pattern))
    return sorted(csv_files)


def find_latest_raw_file(raw_folder: Path, prefix: str = "ethereum_") -> Path | None:
    """Find the most recent raw events CSV file with given prefix."""
    csv_files = list(raw_folder.glob(f"{prefix}*.csv"))
    if not csv_files:
        return None
    # Sort by modification time, newest first
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return csv_files[0]


def find_ens_file(raw_folder: Path) -> Path | None:
    """Find the ENS names CSV file."""
    ens_path = raw_folder / "ens_names.csv"
    if ens_path.exists():
        return ens_path
    return None


def load_ens_names(ens_path: Path) -> set[str]:
    """Load ENS names mapping and return set of addresses with ENS."""
    ens_addresses = set()
    with open(ens_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            address = row.get("address", "").lower()
            if address:
                ens_addresses.add(address)
    return ens_addresses


def safe_log(amount: float) -> float:
    """Calculate log2 of amount, flooring to 1 to avoid negative results."""
    if amount < 1:
        return 0
    return math.log2(amount)


class TrustCalculator:
    """Calculates local trust scores from raw events."""

    def __init__(self, config: dict, ens_addresses: set[str] | None = None):
        self.config = config
        self.trust_weights = config.get("trust_weights", {})
        self.trust_output = config.get("trust_output", {})
        self.output_config = config.get("output", {})

        self.raw_folder = Path(self.output_config.get("raw_folder", "raw"))
        self.trust_folder = Path(self.trust_output.get("trust_folder", "trust"))
        self.trust_folder.mkdir(parents=True, exist_ok=True)

        # ENS verification multiplier (default 2.0)
        self.ens_multiplier = self.trust_output.get("ens_verification_multiplier", 2.0)

        # ENS verified addresses (receive multiplied incoming trust)
        self.ens_addresses = ens_addresses or set()

        # Trust scores: {(from_address, to_address): score}
        self.trust_edges = defaultdict(float)

    def get_decimals(self, protocol: str) -> int:
        """Get the number of decimals for a protocol/token. Defaults to 18."""
        return PROTOCOL_DECIMALS.get(protocol.lower(), 18)

    def get_event_weight(
        self, event_name: str, protocol: str, is_liquidated: bool = False
    ) -> float:
        """Get the trust weight for an event type within a specific protocol."""
        # Get protocol-specific weights, fall back to empty dict
        protocol_weights = self.trust_weights.get(protocol, {})

        if event_name == "LiquidationCall":
            if is_liquidated:
                return protocol_weights.get("LiquidationCall_liquidated", -1.5)
            else:
                return protocol_weights.get("LiquidationCall_liquidator", 1.2)
        return protocol_weights.get(event_name, 0.0)

    def extract_addresses_from_event(
        self, row: dict
    ) -> list[tuple[str, str, float, str, str, bool]]:
        """
        Extract (from_addr, to_addr, amount, event_name, protocol, is_liquidated) tuples from an event.
        Returns list of trust flow tuples.
        """
        event_name = row.get("event_name", "")
        protocol = row.get("protocol", "").lower()
        contract_address = row.get("contract_address", "").lower()

        flows = []

        # Parse amount - prefer pre-decoded amount field, fallback to raw_data
        amount = 0
        if "amount" in row and row["amount"]:
            try:
                amount = int(row["amount"])
            except (ValueError, TypeError):
                amount = 0
        else:
            # Fallback to parsing raw_data
            raw_data = row.get("raw_data", "0x")
            if raw_data and raw_data != "0x" and len(raw_data) >= 66:
                try:
                    amount = int(raw_data[2:66], 16)
                except (ValueError, IndexError):
                    amount = 0

        # Normalize using protocol-specific decimals
        decimals = self.get_decimals(protocol)
        amount_normalized = amount / (10**decimals)

        if event_name == "Supply":
            # User supplies to protocol
            user = row.get("on_behalf_of", "").lower()
            if user:
                flows.append(
                    (
                        user,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "Withdraw":
            # User withdraws from protocol
            user = row.get("user", "").lower()
            if user:
                flows.append(
                    (
                        contract_address,
                        user,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "Borrow":
            # User borrows from protocol
            user = row.get("on_behalf_of", "").lower()
            if user:
                flows.append(
                    (
                        contract_address,
                        user,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "Repay":
            # User repays to protocol
            user = row.get("user", "").lower()
            if user:
                flows.append(
                    (
                        user,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "LiquidationCall":
            # Liquidator liquidates a user's position
            liquidated_user = row.get("user", "").lower()
            # Liquidator address would need to be extracted from transaction
            # For now, we assign negative trust to the liquidated user
            if liquidated_user:
                flows.append(
                    (
                        contract_address,
                        liquidated_user,
                        amount_normalized,
                        event_name,
                        protocol,
                        True,
                    )
                )

        elif event_name == "Transfer":
            # ERC-20 transfer
            from_addr = row.get("from", "").lower()
            to_addr = row.get("to", "").lower()
            if from_addr and to_addr:
                flows.append(
                    (from_addr, to_addr, amount_normalized, event_name, protocol, False)
                )

        elif event_name == "UniswapV3Swap":
            # Uniswap V3 Swap - sender trusts the pool contract
            sender = row.get("sender", "").lower()
            # Use absolute value of amount0 or amount1 (one is positive, one negative)
            amount0 = 0
            amount1 = 0
            if "amount0" in row and row["amount0"]:
                try:
                    amount0 = abs(int(row["amount0"]))
                except (ValueError, TypeError):
                    amount0 = 0
            if "amount1" in row and row["amount1"]:
                try:
                    amount1 = abs(int(row["amount1"]))
                except (ValueError, TypeError):
                    amount1 = 0
            # Use the larger amount (the one being sold)
            swap_amount = max(amount0, amount1)
            # Normalize (assume 18 decimals for most swaps)
            amount_normalized = swap_amount / 1e18
            if sender:
                # Sender trusts the pool contract
                flows.append(
                    (
                        sender,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        return flows

    def process_events(self, csv_path: Path) -> int:
        """Process events from a CSV file and calculate trust scores."""
        print(f"  Processing: {csv_path.name}")

        # First pass: aggregate amounts per (from, to, event, protocol, is_liquidated)
        # Key: (from_addr, to_addr, event_name, protocol, is_liquidated) -> total amount
        aggregated_amounts: dict[tuple[str, str, str, str, bool], float] = defaultdict(
            float
        )

        event_count = 0
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                event_count += 1
                flows = self.extract_addresses_from_event(row)

                for (
                    from_addr,
                    to_addr,
                    amount,
                    event_name,
                    protocol,
                    is_liquidated,
                ) in flows:
                    if not from_addr or not to_addr:
                        continue

                    # Aggregate amounts per (from, to, event, protocol, is_liquidated)
                    agg_key = (from_addr, to_addr, event_name, protocol, is_liquidated)
                    aggregated_amounts[agg_key] += amount

                if event_count % 10000 == 0:
                    print(f"    Processed {event_count:,} events...", end="\r")

        print(f"    Processed {event_count:,} events total" + " " * 20)

        # Second pass: calculate trust from aggregated amounts
        print(
            f"    Calculating trust from {len(aggregated_amounts):,} aggregated pairs..."
        )

        for (
            from_addr,
            to_addr,
            event_name,
            protocol,
            is_liquidated,
        ), total_amount in aggregated_amounts.items():
            weight = self.get_event_weight(event_name, protocol, is_liquidated)
            log_amount = safe_log(total_amount)

            # Calculate trust contribution
            trust_delta = log_amount * weight

            # Apply ENS verification multiplier for incoming trust
            # If the recipient (to_addr) has an ENS name, they get multiplied incoming trust
            if to_addr in self.ens_addresses:
                trust_delta *= self.ens_multiplier

            # Add to edge trust (will floor to 0 later)
            edge_key = (from_addr, to_addr)
            self.trust_edges[edge_key] += trust_delta

        return event_count

    def floor_trust_scores(self):
        """Floor all trust scores to 0 (no negative trust)."""
        floored_count = 0
        for edge_key in self.trust_edges:
            if self.trust_edges[edge_key] < 0:
                self.trust_edges[edge_key] = 0
                floored_count += 1
        print(f"  Floored {floored_count:,} negative trust edges to 0")

    def save_trust_edges(self, output_path: Path):
        """Save trust edges to CSV file."""
        print(f"  Saving trust edges to: {output_path.name}")

        # Filter out zero-trust edges
        non_zero_edges = {k: v for k, v in self.trust_edges.items() if v > 0}

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["i", "j", "v"])

            for (from_addr, to_addr), score in sorted(
                non_zero_edges.items(), key=lambda x: -x[1]
            ):
                writer.writerow([from_addr, to_addr, f"{score:.6f}"])

        print(f"  ✓ Saved {len(non_zero_edges):,} trust edges")

    def print_stats(self):
        """Print statistics."""
        print()
        print("─" * 79)
        print("Statistics")
        print("─" * 79)

        non_zero_edges = sum(1 for v in self.trust_edges.values() if v > 0)
        total_trust = sum(max(0, v) for v in self.trust_edges.values())

        # Count ENS-verified recipients in trust edges
        ens_recipients = set()
        for from_addr, to_addr in self.trust_edges.keys():
            if to_addr in self.ens_addresses:
                ens_recipients.add(to_addr)

        print(f"  Total trust edges: {len(self.trust_edges):,}")
        print(f"  Non-zero trust edges: {non_zero_edges:,}")
        print(f"  Total trust score: {total_trust:,.2f}")
        print(f"  ENS addresses in dataset: {len(self.ens_addresses):,}")
        print(f"  ENS-verified recipients: {len(ens_recipients):,}")
        print(f"  ENS multiplier: {self.ens_multiplier}x")

        if non_zero_edges > 0:
            avg_trust = total_trust / non_zero_edges
            print(f"  Average trust per edge: {avg_trust:.4f}")


def main():
    parser = argparse.ArgumentParser(description="ChainRank Local Trust Calculator")
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config.toml file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV file (default: all files for year/month from config)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to process (default: from config)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Month to process (default: from config)",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default="ethereum",
        help="Chain to process (default: ethereum)",
    )
    parser.add_argument(
        "--ens",
        type=str,
        default=None,
        help="Path to ENS names CSV file (default: latest ens_names_*.csv in raw folder)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(config_path)

    print("═" * 79)
    print("ChainRank Local Trust Calculator")
    print("═" * 79)

    # Load ENS names
    raw_folder = Path(config.get("output", {}).get("raw_folder", "raw"))

    if args.ens:
        ens_path = Path(args.ens)
    else:
        ens_path = find_ens_file(raw_folder)

    ens_addresses = set()
    if ens_path and ens_path.exists():
        print(f"ENS file: {ens_path}")
        ens_addresses = load_ens_names(ens_path)
        print(f"Loaded {len(ens_addresses):,} ENS-verified addresses")
    else:
        print("No ENS file found - proceeding without ENS verification multiplier")

    # Get year/month from args or config
    year = args.year or config.get("year")
    month = args.month or config.get("month")

    # Initialize calculator
    calculator = TrustCalculator(config, ens_addresses)

    # Get chain
    chain = args.chain.lower()

    # Find input files
    if args.input:
        input_paths = [Path(args.input)]
    elif year and month:
        input_paths = find_raw_files_by_chain_year_month(
            calculator.raw_folder, chain, year, month
        )
    else:
        # Fallback to latest file
        latest = find_latest_raw_file(calculator.raw_folder, prefix=f"{chain}_")
        input_paths = [latest] if latest else []

    if not input_paths:
        print(f"Error: No input files found")
        return 1

    print(f"Chain: {chain}")
    print(f"Year: {year}, Month: {month}")
    print(f"Input files: {len(input_paths)}")
    for p in input_paths:
        print(f"  - {p.name}")
    print(f"Trust weights: {calculator.trust_weights}")
    print()

    # Process events from all files
    print("─" * 79)
    print("Processing Events")
    print("─" * 79)
    for input_path in input_paths:
        calculator.process_events(input_path)

    # Floor negative scores
    print()
    print("─" * 79)
    print("Calculating Trust Scores")
    print("─" * 79)
    calculator.floor_trust_scores()

    # Save outputs
    print()
    print("─" * 79)
    print("Saving Results")
    print("─" * 79)

    edges_path = calculator.trust_folder / f"{chain}.csv"

    calculator.save_trust_edges(edges_path)

    # Print statistics
    calculator.print_stats()

    print()
    print("═" * 79)
    print("✓ Done!")
    print("═" * 79)

    return 0


if __name__ == "__main__":
    exit(main())
