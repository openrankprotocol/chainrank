#!/usr/bin/env python3
"""
ChainRank ENS Event Fetcher
═══════════════════════════════════════════════════════════════════════════════
Downloads ENS registration events from the ENS contracts to build address->name mapping.

Fetches NameRegistered events from:
- ETH Registrar Controller (new registrations)
- Old ETH Registrar Controller (legacy registrations)

Usage:
    python fetch_ens.py --config config.toml
    python fetch_ens.py --config config.toml --year 2025

Environment Variables (in .env file):
    RPC_ETHEREUM=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
"""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import toml
from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import BlockNotFound

load_dotenv()


# ENS Contract Addresses
ENS_CONTRACTS = {
    # ETH Registrar Controller (current) - deployed January 2023
    "eth_registrar_controller": {
        "address": "0x253553366Da8546fC250F225fe3d25d0C782303b",
        "deployed_year": 2023,
    },
    # Old ETH Registrar Controller - deployed May 2019
    "eth_registrar_controller_old": {
        "address": "0x283Af0B28c62C092C9727F1Ee09c02CA627EB7F5",
        "deployed_year": 2019,
    },
}

# ENS Event Signatures
# NameRegistered(string,bytes32,address,uint256,uint256,uint256) for new controller
# NameRegistered(string,bytes32,address,uint256,uint256) for old controller
ENS_EVENT_SIGNATURES = {
    # NameRegistered(string name, bytes32 indexed label, address indexed owner, uint256 baseCost, uint256 premium, uint256 expires)
    "NameRegistered": "0x69e37f151eb98a09618ddaa80c8cfaf1ce5996867c489f45b555b412271ebf27",
    # NameRegistered (old controller): NameRegistered(string name, bytes32 indexed label, address indexed owner, uint256 cost, uint256 expires)
    "NameRegisteredOld": "0xca6abbe9d7f11422cb6ca7629fbf6fe9efb1c621f71ce8f02b9f2a230097404f",
}


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "r") as f:
        return toml.load(f)


def get_rpc_endpoint(chain_name: str) -> str:
    """Get RPC endpoint from environment variable."""
    env_var = f"RPC_{chain_name.upper()}"
    endpoint = os.getenv(env_var)
    if not endpoint:
        raise ValueError(f"Missing environment variable: {env_var}")
    return endpoint


def get_block_by_timestamp(w3: Web3, target_timestamp: int) -> int:
    """Binary search to find block number closest to target timestamp."""
    latest_block = w3.eth.block_number
    latest_timestamp = w3.eth.get_block(latest_block)["timestamp"]

    if target_timestamp >= latest_timestamp:
        return latest_block

    # Estimate starting point (assuming ~12s block time)
    blocks_back = (latest_timestamp - target_timestamp) // 12
    low = max(0, latest_block - blocks_back - 10000)
    high = latest_block

    while low < high:
        mid = (low + high) // 2
        try:
            mid_timestamp = w3.eth.get_block(mid)["timestamp"]
        except BlockNotFound:
            high = mid - 1
            continue

        if mid_timestamp < target_timestamp:
            low = mid + 1
        else:
            high = mid

    return low


class ENSEventFetcher:
    """Fetches ENS events from Ethereum."""

    def __init__(self, config: dict):
        self.config = config
        self.output_config = config.get("output", {})
        self.indexer_config = config.get("indexer", {})

        self.raw_folder = Path(self.output_config.get("raw_folder", "raw"))
        self.raw_folder.mkdir(parents=True, exist_ok=True)

        self.batch_size = self.indexer_config.get("batch_size", 2000)
        self.max_retries = self.indexer_config.get("max_retries", 3)
        self.retry_delay = self.indexer_config.get("retry_delay_seconds", 1)
        self.rate_limit = self.indexer_config.get("rate_limit_per_second", 10)
        self.parallel_batches = self.indexer_config.get("parallel_batches", 10)

        self._last_request_time = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        min_interval = 1.0 / self.rate_limit
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _get_web3(self) -> Web3:
        """Get a new Web3 instance for thread-safe access."""
        return Web3(Web3.HTTPProvider(get_rpc_endpoint("ethereum")))

    def _fetch_logs_batch(
        self,
        w3: Web3,
        contract_address: str,
        topics: list[str],
        from_block: int,
        to_block: int,
    ) -> list[dict]:
        """Fetch logs for a batch of blocks with retry logic."""
        for attempt in range(self.max_retries):
            try:
                logs = w3.eth.get_logs(
                    {
                        "address": Web3.to_checksum_address(contract_address),
                        "topics": [topics],
                        "fromBlock": from_block,
                        "toBlock": to_block,
                    }
                )
                return logs
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  Retry {attempt + 1}/{self.max_retries} after error: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"  Failed after {self.max_retries} attempts: {e}")
                    return []
        return []

    def _fetch_batch_task(
        self,
        contract_address: str,
        event_topics: list[str],
        batch_start: int,
        batch_end: int,
        batch_num: int,
    ) -> tuple[int, int, list[dict]]:
        """Task to fetch a single batch. Returns (batch_num, batch_end, logs)."""
        w3 = self._get_web3()
        logs = self._fetch_logs_batch(
            w3, contract_address, event_topics, batch_start, batch_end
        )
        return (batch_num, batch_end, logs)

    def _process_logs(self, logs: list[dict], contract_name: str) -> list[dict]:
        """Process logs and extract ENS registration events."""
        events = []
        for log in logs:
            topic0 = (
                log["topics"][0].hex()
                if isinstance(log["topics"][0], bytes)
                else log["topics"][0]
            )
            if not topic0.startswith("0x"):
                topic0 = "0x" + topic0

            # Extract owner address from topic (indexed parameter)
            owner = None
            if len(log["topics"]) >= 3:
                owner = self._extract_address(log["topics"][2])

            # Decode name from data
            name = self._decode_ens_name(log["data"])

            if owner and name:
                event_record = {
                    "block_number": log["blockNumber"],
                    "transaction_hash": (
                        log["transactionHash"].hex()
                        if isinstance(log["transactionHash"], bytes)
                        else log["transactionHash"]
                    ),
                    "owner": owner,
                    "name": f"{name}.eth",
                    "contract": contract_name,
                }
                events.append(event_record)
        return events

    def _decode_ens_name(self, data: bytes) -> str | None:
        """Decode ENS name from event data."""
        try:
            data_hex = data.hex() if isinstance(data, bytes) else data
            if data_hex.startswith("0x"):
                data_hex = data_hex[2:]

            # The name is encoded as: offset (32 bytes) + length (32 bytes) + name (variable)
            # Skip first 32 bytes (offset), read length from next 32 bytes
            if len(data_hex) < 128:  # Need at least offset + length + some data
                return None

            # Get the offset to the string data
            offset = int(data_hex[0:64], 16) * 2  # Convert to hex char position

            # Get string length from the offset position
            length_start = offset
            length = int(data_hex[length_start : length_start + 64], 16)

            # Get the actual string data
            name_start = length_start + 64
            name_hex = data_hex[name_start : name_start + length * 2]
            name = bytes.fromhex(name_hex).decode("utf-8", errors="ignore")

            return name if name else None
        except Exception:
            return None

    def _extract_address(self, topic) -> str:
        """Extract address from topic."""
        if isinstance(topic, bytes):
            return "0x" + topic.hex()[-40:]
        return "0x" + topic[-40:]

    def fetch_ens_events(self, from_block: int, to_block: int, year: int) -> list[dict]:
        """Fetch ENS registration events."""
        all_events = []

        # Event topics to fetch
        event_topics = [
            ENS_EVENT_SIGNATURES["NameRegistered"],
            ENS_EVENT_SIGNATURES["NameRegisteredOld"],
        ]

        # Filter contracts based on deployment year
        contracts = []
        for name, info in ENS_CONTRACTS.items():
            if year >= info["deployed_year"]:
                contracts.append((name, info["address"]))

        if not contracts:
            print(f"  No contracts deployed for year {year}")
            return all_events

        for contract_name, contract_address in contracts:
            print(f"\n  Contract: {contract_name}")
            print(f"  Address: {contract_address}")

            total_blocks = to_block - from_block + 1
            num_batches = (total_blocks + self.batch_size - 1) // self.batch_size
            print(
                f"  Block range: {from_block:,} → {to_block:,} ({total_blocks:,} blocks, {num_batches} batches)"
            )
            print(f"  Parallel batches: {self.parallel_batches}")

            # Build list of all batches
            batches = []
            current_block = from_block
            batch_num = 0
            while current_block <= to_block:
                batch_end = min(current_block + self.batch_size - 1, to_block)
                batches.append((batch_num, current_block, batch_end))
                batch_num += 1
                current_block = batch_end + 1

            # Process batches in parallel
            start_time = time.time()
            completed_batches = 0
            total_events = 0
            contract_events = []

            with ThreadPoolExecutor(max_workers=self.parallel_batches) as executor:
                # Submit all batch tasks
                futures = {
                    executor.submit(
                        self._fetch_batch_task,
                        contract_address,
                        event_topics,
                        batch_start,
                        batch_end,
                        batch_num,
                    ): batch_num
                    for batch_num, batch_start, batch_end in batches
                }

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        batch_num, batch_end, logs = future.result()
                        completed_batches += 1

                        # Process logs into events
                        events = self._process_logs(logs, contract_name)
                        contract_events.extend(events)
                        total_events += len(logs)

                        # Update progress
                        progress = completed_batches / num_batches * 100
                        elapsed = time.time() - start_time
                        batches_per_sec = completed_batches / max(1, elapsed)
                        eta_seconds = (num_batches - completed_batches) / max(
                            1, batches_per_sec
                        )

                        print(
                            f"    Batch {completed_batches}/{num_batches} ({progress:.1f}%) | "
                            f"{len(contract_events):,} names | {batches_per_sec:.1f} batch/s | ETA: {eta_seconds:.0f}s"
                            + " "
                            * 10,
                            end="\r",
                        )
                    except Exception as e:
                        print(f"\n    Error in batch {futures[future]}: {e}")

            # Sort events by block number to maintain order
            contract_events.sort(key=lambda x: x["block_number"])
            all_events.extend(contract_events)

            elapsed = time.time() - start_time
            print(f"\n  ✓ Completed: {len(contract_events):,} events in {elapsed:.1f}s")

        return all_events

    def build_address_mapping(self, events: list[dict]) -> dict[str, str]:
        """Build address -> ENS name mapping (latest registration wins)."""
        # Sort by block number to get latest registration
        sorted_events = sorted(events, key=lambda x: x["block_number"])

        mapping = {}
        for event in sorted_events:
            mapping[event["owner"].lower()] = event["name"]

        return mapping

    def save_mapping(self, mapping: dict[str, str], output_path: Path):
        """Save address -> ENS name mapping to CSV."""
        print(f"  Saving mapping to: {output_path.name}")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["address", "ens_name"])

            for addr, name in sorted(mapping.items(), key=lambda x: x[1]):
                writer.writerow([addr, name])

        print(f"  ✓ Saved {len(mapping):,} address -> ENS mappings")


def main():
    parser = argparse.ArgumentParser(description="ChainRank ENS Event Fetcher")
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config.toml file"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to fetch data for (e.g., 2025). Defaults to current year.",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(config_path)

    print("═" * 79)
    print("ChainRank ENS Event Fetcher")
    print("═" * 79)

    # Initialize fetcher
    fetcher = ENSEventFetcher(config)

    # Calculate block range
    print()
    print("─" * 79)
    print("Calculating Block Range")
    print("─" * 79)

    year = args.year if args.year else datetime.now(timezone.utc).year
    start_time = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    # If end_time is in the future, use current time
    now = datetime.now(timezone.utc)
    if end_time > now:
        end_time = now

    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    print(f"  Year: {year}")
    print(f"  Time window: {start_time.isoformat()} to {end_time.isoformat()}")

    w3 = fetcher._get_web3()
    from_block = get_block_by_timestamp(w3, start_timestamp)
    to_block = get_block_by_timestamp(w3, end_timestamp)

    print(f"  Block range: {from_block:,} to {to_block:,}")

    # Fetch events
    print()
    print("─" * 79)
    print("Fetching ENS Events")
    print("─" * 79)

    events = fetcher.fetch_ens_events(from_block, to_block, year)

    # Save events
    print()
    print("─" * 79)
    print("Saving Results")
    print("─" * 79)

    mapping_path = fetcher.raw_folder / f"ens_names_{year}.csv"

    # Build and save mapping
    mapping = fetcher.build_address_mapping(events)
    fetcher.save_mapping(mapping, mapping_path)

    # Print stats
    print()
    print("─" * 79)
    print("Statistics")
    print("─" * 79)
    print(f"  Total registrations: {len(events):,}")
    print(f"  Unique addresses with ENS: {len(mapping):,}")

    print()
    print("═" * 79)
    print("✓ Done!")
    print("═" * 79)

    return 0


if __name__ == "__main__":
    exit(main())
