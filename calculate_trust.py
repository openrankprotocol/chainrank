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
import json
import math
from collections import defaultdict
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


def find_all_raw_files_by_chain(raw_folder: Path, chain: str) -> list[Path]:
    """Find all raw event CSV files for a specific chain (excluding ens_names and cex_addresses)."""
    pattern = f"{chain}_*.csv"
    csv_files = [
        f
        for f in raw_folder.glob(pattern)
        if not f.name.startswith("ens_names") and not f.name.startswith("cex_")
    ]
    return sorted(csv_files)


def find_latest_raw_file(raw_folder: Path, prefix: str = "ethereum_") -> Path | None:
    """Find the most recent raw events CSV file with given prefix."""
    csv_files = list(raw_folder.glob(f"{prefix}*.csv"))
    if not csv_files:
        return None
    # Sort by modification time, newest first
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return csv_files[0]


def find_ens_files(raw_folder: Path) -> list[Path]:
    """Find all ENS names CSV files (ens_names_*.csv)."""
    ens_files = sorted(raw_folder.glob("ens_names_*.csv"))
    return ens_files


def load_ens_names(ens_paths: list[Path]) -> set[str]:
    """Load ENS names mapping from multiple files and return set of addresses with ENS."""
    ens_addresses = set()
    for ens_path in ens_paths:
        with open(ens_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                address = row.get("address", "").lower()
                if address:
                    ens_addresses.add(address)
    return ens_addresses


def load_fc_addresses(raw_folder: Path) -> set[str]:
    """Load Farcaster verified addresses from fc_addresses.csv."""
    fc_path = raw_folder / "fc_addresses.csv"
    fc_addresses = set()
    if not fc_path.exists():
        return fc_addresses
    with open(fc_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addresses_json = row.get("verified_addresses", "[]")
            try:
                addresses = json.loads(addresses_json)
                for addr in addresses:
                    # Only include 20-byte Ethereum addresses (42 chars with 0x)
                    if addr and len(addr) == 42 and addr.startswith("0x"):
                        fc_addresses.add(addr.lower())
            except json.JSONDecodeError:
                pass
    return fc_addresses


def safe_log(amount: float) -> float:
    """Calculate log2 of amount, flooring to 1 to avoid negative results."""
    if amount < 1:
        return 0
    return math.log2(amount)


class TrustCalculator:
    """Calculates local trust scores from raw events."""

    def __init__(
        self,
        config: dict,
        ens_addresses: set[str] | None = None,
        fc_addresses: set[str] | None = None,
    ):
        self.config = config
        self.trust_weights = config.get("trust_weights", {})
        self.trust_output = config.get("trust_output", {})
        self.output_config = config.get("output", {})

        self.raw_folder = Path(self.output_config.get("raw_folder", "raw"))
        self.trust_folder = Path(self.trust_output.get("trust_folder", "trust"))
        self.trust_folder.mkdir(parents=True, exist_ok=True)

        # ENS verification multiplier (default 2.0)
        self.ens_multiplier = self.trust_output.get("ens_verification_multiplier", 2.0)

        # Farcaster verification multiplier (default 2.0)
        self.fc_multiplier = self.trust_output.get("fc_verification_multiplier", 2.0)

        # ENS verified addresses (receive multiplied incoming trust)
        self.ens_addresses = ens_addresses or set()

        # Farcaster verified addresses (receive multiplied incoming trust)
        self.fc_addresses = fc_addresses or set()

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

        # Compound V2 Events
        elif event_name == "Mint":
            # User deposits underlying asset, receives cTokens
            minter = row.get("minter", "").lower()
            mint_amount = 0
            if "mint_amount" in row and row["mint_amount"]:
                try:
                    mint_amount = int(row["mint_amount"])
                except (ValueError, TypeError):
                    mint_amount = 0
            amount_normalized = mint_amount / (10**decimals)
            if minter:
                flows.append(
                    (
                        minter,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "Redeem":
            # User redeems cTokens for underlying asset
            redeemer = row.get("redeemer", "").lower()
            redeem_amount = 0
            if "redeem_amount" in row and row["redeem_amount"]:
                try:
                    redeem_amount = int(row["redeem_amount"])
                except (ValueError, TypeError):
                    redeem_amount = 0
            amount_normalized = redeem_amount / (10**decimals)
            if redeemer:
                flows.append(
                    (
                        contract_address,
                        redeemer,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "CompoundBorrow":
            # User borrows from protocol (Compound-style Borrow)
            borrower = row.get("borrower", "").lower()
            borrow_amount = 0
            if "borrow_amount" in row and row["borrow_amount"]:
                try:
                    borrow_amount = int(row["borrow_amount"])
                except (ValueError, TypeError):
                    borrow_amount = 0
            amount_normalized = borrow_amount / (10**decimals)
            if borrower:
                flows.append(
                    (
                        contract_address,
                        borrower,
                        amount_normalized,
                        "CompoundBorrow",
                        protocol,
                        False,
                    )
                )

        elif event_name == "RepayBorrow":
            # User repays borrowed assets
            payer = row.get("payer", "").lower()
            borrower = row.get("borrower", "").lower()
            repay_amount = 0
            if "repay_amount" in row and row["repay_amount"]:
                try:
                    repay_amount = int(row["repay_amount"])
                except (ValueError, TypeError):
                    repay_amount = 0
            amount_normalized = repay_amount / (10**decimals)
            # Credit goes to the payer (who may be repaying for someone else)
            if payer:
                flows.append(
                    (
                        payer,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "LiquidateBorrow":
            # Liquidator repays debt and seizes collateral
            liquidator = row.get("liquidator", "").lower()
            borrower = row.get("borrower", "").lower()
            repay_amount = 0
            if "repay_amount" in row and row["repay_amount"]:
                try:
                    repay_amount = int(row["repay_amount"])
                except (ValueError, TypeError):
                    repay_amount = 0
            amount_normalized = repay_amount / (10**decimals)
            # Liquidator gets positive trust for maintaining protocol health
            if liquidator:
                flows.append(
                    (
                        liquidator,
                        contract_address,
                        amount_normalized,
                        "LiquidateBorrow_liquidator",
                        protocol,
                        False,
                    )
                )
            # Borrower gets negative trust for being liquidated
            if borrower:
                flows.append(
                    (
                        contract_address,
                        borrower,
                        amount_normalized,
                        "LiquidateBorrow_liquidated",
                        protocol,
                        True,
                    )
                )

        elif event_name == "DistributedSupplierComp":
            # COMP rewards distributed to supplier
            supplier = row.get("supplier", "").lower()
            comp_delta = 0
            if "comp_delta" in row and row["comp_delta"]:
                try:
                    comp_delta = int(row["comp_delta"])
                except (ValueError, TypeError):
                    comp_delta = 0
            # COMP has 18 decimals
            amount_normalized = comp_delta / 1e18
            if supplier:
                flows.append(
                    (
                        contract_address,
                        supplier,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "DistributedBorrowerComp":
            # COMP rewards distributed to borrower
            borrower = row.get("borrower", "").lower()
            comp_delta = 0
            if "comp_delta" in row and row["comp_delta"]:
                try:
                    comp_delta = int(row["comp_delta"])
                except (ValueError, TypeError):
                    comp_delta = 0
            # COMP has 18 decimals
            amount_normalized = comp_delta / 1e18
            if borrower:
                flows.append(
                    (
                        contract_address,
                        borrower,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        # Morpho Blue Events
        elif event_name == "MorphoSupply":
            # User supplies assets to Morpho Blue
            on_behalf_of = row.get("on_behalf_of", "").lower()
            assets = 0
            if "assets" in row and row["assets"]:
                try:
                    assets = int(row["assets"])
                except (ValueError, TypeError):
                    assets = 0
            amount_normalized = assets / (10**decimals)
            if on_behalf_of:
                flows.append(
                    (
                        on_behalf_of,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "MorphoWithdraw":
            # User withdraws from Morpho Blue
            receiver = row.get("receiver", "").lower()
            assets = 0
            if "assets" in row and row["assets"]:
                try:
                    assets = int(row["assets"])
                except (ValueError, TypeError):
                    assets = 0
            amount_normalized = assets / (10**decimals)
            if receiver:
                flows.append(
                    (
                        contract_address,
                        receiver,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "MorphoBorrow":
            # User borrows from Morpho Blue
            receiver = row.get("receiver", "").lower()
            assets = 0
            if "assets" in row and row["assets"]:
                try:
                    assets = int(row["assets"])
                except (ValueError, TypeError):
                    assets = 0
            amount_normalized = assets / (10**decimals)
            if receiver:
                flows.append(
                    (
                        contract_address,
                        receiver,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "MorphoRepay":
            # User repays to Morpho Blue
            caller = row.get("caller", "").lower()
            assets = 0
            if "assets" in row and row["assets"]:
                try:
                    assets = int(row["assets"])
                except (ValueError, TypeError):
                    assets = 0
            amount_normalized = assets / (10**decimals)
            if caller:
                flows.append(
                    (
                        caller,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "MorphoLiquidate":
            # Morpho Blue liquidation
            liquidator = row.get("liquidator", "").lower()
            borrower = row.get("borrower", "").lower()
            repaid_assets = 0
            if "repaid_assets" in row and row["repaid_assets"]:
                try:
                    repaid_assets = int(row["repaid_assets"])
                except (ValueError, TypeError):
                    repaid_assets = 0
            amount_normalized = repaid_assets / (10**decimals)
            # Liquidator gets positive trust
            if liquidator:
                flows.append(
                    (
                        liquidator,
                        contract_address,
                        amount_normalized,
                        "MorphoLiquidate_liquidator",
                        protocol,
                        False,
                    )
                )
            # Borrower gets negative trust
            if borrower:
                flows.append(
                    (
                        contract_address,
                        borrower,
                        amount_normalized,
                        "MorphoLiquidate_liquidated",
                        protocol,
                        True,
                    )
                )

        # Fluid Liquidity Events
        elif event_name == "FluidOperate":
            # Fluid combined supply/borrow operation
            user = row.get("user", "").lower()
            supply_amount = 0
            borrow_amount = 0
            if "supply_amount" in row and row["supply_amount"]:
                try:
                    supply_amount = int(row["supply_amount"])
                except (ValueError, TypeError):
                    supply_amount = 0
            if "borrow_amount" in row and row["borrow_amount"]:
                try:
                    borrow_amount = int(row["borrow_amount"])
                except (ValueError, TypeError):
                    borrow_amount = 0
            # Positive supply_amount means deposit, negative means withdraw
            # Positive borrow_amount means borrow, negative means repay
            if user:
                if supply_amount > 0:
                    # User depositing
                    amount_normalized = supply_amount / (10**decimals)
                    flows.append(
                        (
                            user,
                            contract_address,
                            amount_normalized,
                            "FluidOperate_supply",
                            protocol,
                            False,
                        )
                    )
                elif supply_amount < 0:
                    # User withdrawing
                    amount_normalized = abs(supply_amount) / (10**decimals)
                    flows.append(
                        (
                            contract_address,
                            user,
                            amount_normalized,
                            "FluidOperate_withdraw",
                            protocol,
                            False,
                        )
                    )
                if borrow_amount < 0:
                    # User repaying (negative borrow = repay)
                    amount_normalized = abs(borrow_amount) / (10**decimals)
                    flows.append(
                        (
                            user,
                            contract_address,
                            amount_normalized,
                            "FluidOperate_repay",
                            protocol,
                            False,
                        )
                    )
                elif borrow_amount > 0:
                    # User borrowing
                    amount_normalized = borrow_amount / (10**decimals)
                    flows.append(
                        (
                            contract_address,
                            user,
                            amount_normalized,
                            "FluidOperate_borrow",
                            protocol,
                            False,
                        )
                    )

        # DEX Aggregator Events
        elif event_name == "OneInchSwapped":
            # 1inch OrderFilled - maker is the user who filled the order
            maker = row.get("maker", "").lower()
            remaining_amount = 0
            if "remaining_amount" in row and row["remaining_amount"]:
                try:
                    remaining_amount = int(row["remaining_amount"])
                except (ValueError, TypeError):
                    remaining_amount = 0
            amount_normalized = remaining_amount / (10**decimals)
            if maker:
                flows.append(
                    (
                        maker,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "ZeroExTransformedERC20":
            # 0x swap - taker is the user
            taker = row.get("taker", "").lower()
            input_amount = 0
            if "input_amount" in row and row["input_amount"]:
                try:
                    input_amount = int(row["input_amount"])
                except (ValueError, TypeError):
                    input_amount = 0
            amount_normalized = input_amount / (10**decimals)
            if taker:
                flows.append(
                    (
                        taker,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "ParaswapSwapped":
            # Paraswap swap - initiator is the user
            initiator = row.get("initiator", "").lower()
            src_amount = 0
            if "src_amount" in row and row["src_amount"]:
                try:
                    src_amount = int(row["src_amount"])
                except (ValueError, TypeError):
                    src_amount = 0
            amount_normalized = src_amount / (10**decimals)
            if initiator:
                flows.append(
                    (
                        initiator,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
                    )
                )

        elif event_name == "CowTrade":
            # CoW Protocol trade - owner is the user
            owner = row.get("owner", "").lower()
            sell_amount = 0
            if "sell_amount" in row and row["sell_amount"]:
                try:
                    sell_amount = int(row["sell_amount"])
                except (ValueError, TypeError):
                    sell_amount = 0
            amount_normalized = sell_amount / (10**decimals)
            if owner:
                flows.append(
                    (
                        owner,
                        contract_address,
                        amount_normalized,
                        event_name,
                        protocol,
                        False,
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

            # Apply verification multipliers for incoming trust (they stack)
            # If the recipient (to_addr) has an ENS name, they get multiplied incoming trust
            if to_addr in self.ens_addresses:
                trust_delta *= self.ens_multiplier
            # If the recipient has a Farcaster verified address, apply FC multiplier
            if to_addr in self.fc_addresses:
                trust_delta *= self.fc_multiplier

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
        print(f"  FC addresses in dataset: {len(self.fc_addresses):,}")
        print(f"  FC multiplier: {self.fc_multiplier}x")

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
        help="Path to input CSV file (default: all raw files for chain in raw folder)",
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
        help="Path to ENS names CSV file (default: all ens_names_*.csv files in raw folder)",
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
        ens_paths = [Path(args.ens)]
    else:
        ens_paths = find_ens_files(raw_folder)

    if ens_paths:
        print(f"ENS files: {len(ens_paths)} file(s)")
        for p in ens_paths:
            print(f"  - {p.name}")
        ens_addresses = load_ens_names(ens_paths)
        print(f"Loaded {len(ens_addresses):,} ENS-verified addresses")
    else:
        print("No ENS file found - proceeding without ENS verification multiplier")
        ens_addresses = set()

    # Load Farcaster addresses
    fc_addresses = load_fc_addresses(raw_folder)
    if fc_addresses:
        print(f"Loaded {len(fc_addresses):,} Farcaster-verified addresses")
    else:
        print("No Farcaster addresses found")

    # Initialize calculator
    calculator = TrustCalculator(config, ens_addresses, fc_addresses)

    # Get chain
    chain = args.chain.lower()

    # Find input files
    if args.input:
        input_paths = [Path(args.input)]
    else:
        # Use all raw files for the chain
        input_paths = find_all_raw_files_by_chain(calculator.raw_folder, chain)

    if not input_paths:
        print(f"Error: No input files found")
        return 1

    print(f"Chain: {chain}")
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
