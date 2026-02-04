#!/usr/bin/env python3
"""
ChainRank Score Analyzer

Analyzes top addresses from scores files and enriches them with:
- Net worth in USD (from Alchemy token balances + prices)
- Wallet volume in USD (from Alchemy transfers + prices)
- Transaction count (from Alchemy)
- Last active timestamp (from Alchemy)
- First active timestamp (from Alchemy)
- ENS name (from local raw/dune_ens_names.csv)
- Farcaster username (from local raw/fc_addresses.csv)
- World ID verification status (from local raw/wrld_addresses.csv)

SETUP:
1. Create a .env file with RPC_ETHEREUM and/or RPC_BASE (Alchemy endpoints)
"""

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

# CEX addresses cache (loaded from file)
_cex_addresses_cache = None


def load_cex_addresses(raw_dir: Path) -> set[str]:
    """Load CEX addresses from raw/cex_addresses.csv file."""
    global _cex_addresses_cache

    if _cex_addresses_cache is not None:
        return _cex_addresses_cache

    cex_file = raw_dir / "cex_addresses.csv"
    cex_addresses = set()

    # Always exclude zero address
    cex_addresses.add("0x0000000000000000000000000000000000000000")

    if not cex_file.exists():
        print(f"  Warning: CEX addresses file not found: {cex_file}")
        _cex_addresses_cache = cex_addresses
        return cex_addresses

    with open(cex_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row.get("address", "").lower()
            if addr and addr.startswith("0x"):
                cex_addresses.add(addr)

    print(f"  Loaded {len(cex_addresses)} CEX addresses")
    _cex_addresses_cache = cex_addresses
    return cex_addresses


# Major token addresses by chain (lowercase)
MAJOR_TOKENS = {
    "ethereum": {
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH", 18),
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC", 6),
        "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT", 6),
        "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI", 18),
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": ("WBTC", 8),
    },
    "base": {
        "0x4200000000000000000000000000000000000006": ("WETH", 18),
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": ("USDC", 6),
        "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2": ("USDT", 6),
        "0x50c5725949a6f0c72e6c4a641f24049a917db0cb": ("DAI", 18),
        "0x0555e30da8f98308edb960aa94c0db47230d2b9c": ("WBTC", 8),
        "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": ("cbBTC", 8),
    },
}

# CoinGecko IDs for price lookup
COINGECKO_IDS = {
    "ETH": "ethereum",
    "WETH": "ethereum",
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "WBTC": "wrapped-bitcoin",
    "cbBTC": "wrapped-bitcoin",
}

# Cache for token prices
_price_cache = {}


def get_rpc_endpoint(chain: str) -> str | None:
    """Get RPC endpoint for chain from environment."""
    env_var = f"RPC_{chain.upper()}"
    return os.getenv(env_var)


def get_alchemy_api_key(chain: str) -> str | None:
    """Extract Alchemy API key from RPC URL."""
    rpc_url = get_rpc_endpoint(chain)
    if rpc_url and "alchemy.com" in rpc_url:
        # URL format: https://{network}.g.alchemy.com/v2/{api_key}
        parts = rpc_url.rstrip("/").split("/")
        if len(parts) >= 1:
            return parts[-1]
    return None


def get_alchemy_base_url(chain: str) -> str | None:
    """Get Alchemy base URL for a chain."""
    rpc_url = get_rpc_endpoint(chain)
    if rpc_url and "alchemy.com" in rpc_url:
        # Convert RPC URL to base URL
        # From: https://eth-mainnet.g.alchemy.com/v2/{key}
        # To: https://eth-mainnet.g.alchemy.com
        parts = rpc_url.split("/v2/")
        if len(parts) >= 1:
            return parts[0]
    return None


def fetch_token_prices() -> dict[str, float]:
    """Fetch current USD prices for major tokens from CoinGecko."""
    global _price_cache

    # Return cached prices if fresh (less than 5 minutes old)
    if _price_cache.get("_timestamp", 0) > time.time() - 300:
        return _price_cache

    try:
        ids = ",".join(set(COINGECKO_IDS.values()))
        url = (
            f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
        )
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            prices = {}
            for symbol, cg_id in COINGECKO_IDS.items():
                if cg_id in data and "usd" in data[cg_id]:
                    prices[symbol] = data[cg_id]["usd"]

            prices["_timestamp"] = time.time()
            _price_cache = prices
            return prices
    except Exception as e:
        print(f"  Warning: Could not fetch prices from CoinGecko: {e}")

    # Return stale cache or defaults
    if _price_cache:
        return _price_cache

    # Fallback prices
    return {
        "ETH": 3000,
        "WETH": 3000,
        "USDC": 1,
        "USDT": 1,
        "DAI": 1,
        "WBTC": 60000,
        "cbBTC": 60000,
    }


def is_eoa(w3: Web3, address: str) -> bool:
    """Check if an address is an EOA (not a contract)."""
    try:
        checksum_addr = Web3.to_checksum_address(address)
        code = w3.eth.get_code(checksum_addr)
        # EOAs have no code (empty or 0x)
        return code == b"" or code == b"0x" or len(code) <= 2
    except Exception:
        return False


def is_cex_address(address: str, cex_addresses: set[str]) -> bool:
    """Check if an address is a known CEX address."""
    return address.lower() in cex_addresses


def check_etherscan_label(address: str) -> str | None:
    """Check Etherscan for address label (to identify CEX/exchange addresses)."""
    if not ETHERSCAN_API_KEY:
        return None

    try:
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset=1&sort=asc&apikey={ETHERSCAN_API_KEY}"
        # Note: Etherscan doesn't have a direct label API, but we can check account tags
        # For now, rely on the static list
        return None
    except Exception:
        return None


def load_top_eoas_from_scores(
    scores_dir: Path,
    chain: str,
    raw_dir: Path,
    count: int = 100,
    ens_addresses: set[str] | None = None,
    fc_addresses: set[str] | None = None,
    wrld_addresses: set[str] | None = None,
) -> list[tuple[str, float, int, float]]:
    """Load top EOA addresses from scores file.

    Returns list of (address, normalized_score, original_rank, original_score) tuples for EOAs only.
    Scores are passed through log2() then normalized between 0.0-1.0.
    Excludes contracts, CEX addresses, zero address, and addresses without identity verification.
    """
    scores_file = scores_dir / f"{chain}.csv"

    if not scores_file.exists():
        print(f"  Error: Scores file not found: {scores_file}")
        return []

    # First pass: read all scores to compute min/max for normalization
    print(f"  Reading scores from {scores_file}...")
    all_scores = []
    with open(scores_file, "r") as f:
        reader = csv.DictReader(f)
        for rank, row in enumerate(reader, 1):
            address = row.get("i", "").lower()
            score = float(row.get("v", 0))
            if address and address.startswith("0x") and score > 0:
                # Apply log2 transformation
                log_score = math.log2(
                    score + 1e-10
                )  # Add small epsilon to avoid log(0)
                all_scores.append((address, log_score, rank, score))

    if not all_scores:
        print(f"  Error: No valid scores found")
        return []

    # Normalize log scores to 0.0-1.0
    log_scores = [s[1] for s in all_scores]
    min_log = min(log_scores)
    max_log = max(log_scores)
    log_range = max_log - min_log if max_log > min_log else 1.0

    # Create normalized scores list: (address, normalized_score, original_rank, original_score)
    normalized_scores = []
    for address, log_score, rank, orig_score in all_scores:
        normalized = (log_score - min_log) / log_range
        normalized_scores.append((address, normalized, rank, orig_score))

    print(f"  Loaded {len(normalized_scores)} addresses with normalized scores")

    # Get RPC endpoint for EOA checking
    rpc_url = get_rpc_endpoint(chain)
    if not rpc_url:
        print(f"  Warning: RPC_{chain.upper()} not set, cannot filter EOAs")
        print(f"  Will return top {count} addresses without EOA filtering")
        w3 = None
    else:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            print(f"  Warning: Cannot connect to {chain} RPC, skipping EOA filtering")
            w3 = None

    # Load CEX addresses
    cex_addresses = load_cex_addresses(raw_dir)

    # Identity sets (for filtering)
    ens_set = ens_addresses or set()
    fc_set = fc_addresses or set()
    wrld_set = wrld_addresses or set()
    require_identity = bool(ens_set or fc_set or wrld_set)

    eoas = []
    checked = 0
    contracts_skipped = 0
    cex_skipped = 0
    no_identity_skipped = 0

    if require_identity:
        print(f"  Filtering to addresses with ENS/Farcaster/WorldID...")

    for address, norm_score, orig_rank, orig_score in normalized_scores:
        if len(eoas) >= count:
            break

        checked += 1

        # Skip known CEX addresses and zero address
        if is_cex_address(address, cex_addresses):
            cex_skipped += 1
            continue

        # Skip addresses without identity verification (if filtering enabled)
        if require_identity:
            has_identity = (
                address in ens_set or address in fc_set or address in wrld_set
            )
            if not has_identity:
                no_identity_skipped += 1
                continue

        # Check if EOA
        if w3:
            if is_eoa(w3, address):
                eoas.append((address, norm_score, orig_rank, orig_score))
                if len(eoas) % 10 == 0:
                    print(
                        f"    Found {len(eoas)} EOAs (checked {checked}, skipped {contracts_skipped} contracts, {cex_skipped} CEX, {no_identity_skipped} no identity)..."
                    )
            else:
                contracts_skipped += 1
        else:
            # No RPC, just take top addresses
            eoas.append((address, norm_score, orig_rank, orig_score))

    print(
        f"  Found {len(eoas)} EOAs after checking {checked} addresses ({contracts_skipped} contracts, {cex_skipped} CEX, {no_identity_skipped} no identity skipped)"
    )
    return eoas


def load_ens_names(raw_dir: Path) -> dict[str, str]:
    """Load ENS names from raw directory."""
    ens_file = raw_dir / "dune_ens_names.csv"
    ens_map = {}

    if not ens_file.exists():
        print(f"  Warning: ENS file not found: {ens_file}")
        return ens_map

    with open(ens_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row.get("address", "").lower()
            name = row.get("name", "")
            if addr and name:
                ens_map[addr] = name

    print(f"  Loaded {len(ens_map)} ENS names")
    return ens_map


def load_farcaster_names(raw_dir: Path) -> dict[str, str]:
    """Load Farcaster usernames from raw directory."""
    fc_file = raw_dir / "fc_addresses.csv"
    fc_map = {}

    if not fc_file.exists():
        print(f"  Warning: Farcaster file not found: {fc_file}")
        return fc_map

    with open(fc_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("fname", "")
            verified_addresses = row.get("verified_addresses", "[]")

            try:
                addresses = json.loads(verified_addresses.replace("'", '"'))
                for addr in addresses:
                    addr_lower = addr.lower()
                    if addr_lower.startswith("0x") and len(addr_lower) == 42:
                        if addr_lower not in fc_map:
                            fc_map[addr_lower] = fname
            except (json.JSONDecodeError, TypeError):
                continue

    print(f"  Loaded {len(fc_map)} Farcaster usernames")
    return fc_map


def load_worldcoin_addresses(raw_dir: Path) -> set[str]:
    """Load World ID verified addresses from raw directory."""
    wrld_file = raw_dir / "wrld_addresses.csv"
    wrld_set = set()

    if not wrld_file.exists():
        print(f"  Warning: World ID file not found: {wrld_file}")
        return wrld_set

    with open(wrld_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row.get("address", "").lower()
            if addr:
                wrld_set.add(addr)

    print(f"  Loaded {len(wrld_set)} World ID verified addresses")
    return wrld_set


def format_timestamp(timestamp: int | None) -> str:
    """Convert unix timestamp to 'X days ago' format."""
    if not timestamp:
        return ""
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        days = delta.days

        if days < 0:
            days = abs(days)

        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                return "just now"
            return f"{hours}h ago"
        elif days == 1:
            return "1 day ago"
        elif days < 30:
            return f"{days} days ago"
        elif days < 365:
            months = days // 30
            return f"{months}mo ago"
        else:
            years = days // 365
            return f"{years}y ago"
    except (ValueError, TypeError):
        return ""


def format_currency(value) -> str:
    """Convert a number to human-readable currency format like $1.2k, $100, $0.1."""
    if value is None or value == "":
        return ""
    try:
        usd_value = float(value)
        if usd_value == 0:
            return "$0"

        if abs(usd_value) >= 1_000_000_000:
            return f"${usd_value / 1_000_000_000:.1f}b"
        elif abs(usd_value) >= 1_000_000:
            return f"${usd_value / 1_000_000:.1f}m"
        elif abs(usd_value) >= 1_000:
            return f"${usd_value / 1_000:.1f}k"
        elif abs(usd_value) >= 1:
            return f"${usd_value:.1f}"
        elif abs(usd_value) >= 0.01:
            return f"${usd_value:.2f}"
        else:
            return "<$0.01"
    except (ValueError, TypeError):
        return ""


def fetch_address_stats_alchemy(
    address: str, chain: str, api_key: str, base_url: str, prices: dict[str, float]
) -> dict:
    """Fetch wallet statistics from Alchemy for a single address."""
    stats = {
        "tx_count": 0,
        "net_worth_usd": 0,
        "wallet_volume_usd": 0,
        "first_active": None,
        "last_active": None,
    }

    checksum_addr = Web3.to_checksum_address(address)
    headers = {"accept": "application/json", "content-type": "application/json"}
    rpc_url = f"{base_url}/v2/{api_key}"

    # Get major tokens for this chain
    major_tokens = MAJOR_TOKENS.get(chain, {})

    # Calculate 6 months ago timestamp for volume filtering
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)

    try:
        # 1. Get transaction count
        tx_count_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getTransactionCount",
            "params": [checksum_addr, "latest"],
        }

        resp = requests.post(
            rpc_url, json=tx_count_payload, headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if "result" in data:
                stats["tx_count"] = int(data["result"], 16)

        # 2. Get ETH balance and add to net worth
        eth_balance_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_getBalance",
            "params": [checksum_addr, "latest"],
        }

        resp = requests.post(
            rpc_url, json=eth_balance_payload, headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if "result" in data:
                eth_wei = int(data["result"], 16)
                eth_balance = eth_wei / 10**18
                eth_price = prices.get("ETH", 0)
                stats["net_worth_usd"] += eth_balance * eth_price

        # 3. Get token balances for major tokens
        token_balances_payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "alchemy_getTokenBalances",
            "params": [checksum_addr, list(major_tokens.keys())],
        }

        resp = requests.post(
            rpc_url, json=token_balances_payload, headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if "result" in data and "tokenBalances" in data["result"]:
                for token_data in data["result"]["tokenBalances"]:
                    token_addr = token_data.get("contractAddress", "").lower()
                    balance_hex = token_data.get("tokenBalance", "0x0")

                    if token_addr in major_tokens and balance_hex != "0x0":
                        symbol, decimals = major_tokens[token_addr]
                        try:
                            balance = int(balance_hex, 16) / (10**decimals)
                            price = prices.get(symbol, 0)
                            stats["net_worth_usd"] += balance * price
                        except:
                            pass

        # 4. Get OUTGOING transfers (last 6 months) for volume + last_active
        total_volume_usd = 0

        outgoing_payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromAddress": checksum_addr,
                    "category": ["external", "erc20"],
                    "order": "desc",
                    "maxCount": "0x3e8",  # 1000
                    "withMetadata": True,
                }
            ],
        }

        resp = requests.post(
            rpc_url, json=outgoing_payload, headers=headers, timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            if "result" in data and "transfers" in data["result"]:
                transfers = data["result"]["transfers"]

                # Last active from most recent outgoing transfer
                if transfers:
                    ts_str = transfers[0].get("metadata", {}).get("blockTimestamp", "")
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            stats["last_active"] = int(dt.timestamp())
                        except:
                            pass

                # Sum volume in USD for last 6 months
                for transfer in transfers:
                    ts_str = transfer.get("metadata", {}).get("blockTimestamp", "")
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if dt < six_months_ago:
                                break  # Stop if older than 6 months (desc order)

                            value = transfer.get("value")
                            if value is not None:
                                asset = transfer.get("asset", "").upper()
                                # Map common asset names
                                if asset in ["ETH", "WETH"]:
                                    price = prices.get("ETH", 0)
                                elif asset in prices:
                                    price = prices.get(asset, 0)
                                else:
                                    # Try to identify stablecoins
                                    if asset in ["USDC", "USDT", "DAI", "BUSD", "TUSD"]:
                                        price = 1
                                    else:
                                        price = 0  # Unknown token
                                total_volume_usd += float(value) * price
                        except:
                            pass

        # 5. Get INCOMING transfers (last 6 months) for volume
        incoming_payload = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "toAddress": checksum_addr,
                    "category": ["external", "erc20"],
                    "order": "desc",
                    "maxCount": "0x3e8",  # 1000
                    "withMetadata": True,
                }
            ],
        }

        resp = requests.post(
            rpc_url, json=incoming_payload, headers=headers, timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            if "result" in data and "transfers" in data["result"]:
                transfers = data["result"]["transfers"]

                # Sum volume in USD for last 6 months
                for transfer in transfers:
                    ts_str = transfer.get("metadata", {}).get("blockTimestamp", "")
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if dt < six_months_ago:
                                break  # Stop if older than 6 months (desc order)

                            value = transfer.get("value")
                            if value is not None:
                                asset = transfer.get("asset", "").upper()
                                # Map common asset names
                                if asset in ["ETH", "WETH"]:
                                    price = prices.get("ETH", 0)
                                elif asset in prices:
                                    price = prices.get(asset, 0)
                                else:
                                    # Try to identify stablecoins
                                    if asset in ["USDC", "USDT", "DAI", "BUSD", "TUSD"]:
                                        price = 1
                                    else:
                                        price = 0  # Unknown token
                                total_volume_usd += float(value) * price
                        except:
                            pass

        stats["wallet_volume_usd"] = total_volume_usd

        # 6. Get first active (oldest outgoing transfer)
        first_active_payload = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromAddress": checksum_addr,
                    "category": ["external", "erc20"],
                    "order": "asc",
                    "maxCount": "0x1",
                    "withMetadata": True,
                }
            ],
        }

        resp = requests.post(
            rpc_url, json=first_active_payload, headers=headers, timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            if (
                "result" in data
                and "transfers" in data["result"]
                and data["result"]["transfers"]
            ):
                ts_str = (
                    data["result"]["transfers"][0]
                    .get("metadata", {})
                    .get("blockTimestamp", "")
                )
                if ts_str:
                    try:
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        stats["first_active"] = int(dt.timestamp())
                    except:
                        pass

    except Exception as e:
        # Silently handle errors for individual addresses
        pass

    return stats


def fetch_wallet_stats_from_alchemy(
    addresses: list[str], chain: str
) -> dict[str, dict]:
    """Fetch wallet statistics from Alchemy for given addresses."""
    api_key = get_alchemy_api_key(chain)
    base_url = get_alchemy_base_url(chain)

    if not api_key or not base_url:
        print(f"  Error: Could not extract Alchemy API key from RPC_{chain.upper()}")
        return {}

    if not addresses:
        print("  No addresses to query")
        return {}

    # Fetch token prices first
    print("  Fetching token prices...")
    prices = fetch_token_prices()
    print(f"    ETH: ${prices.get('ETH', 0):,.0f}, BTC: ${prices.get('WBTC', 0):,.0f}")

    all_stats = {}
    completed = 0
    total = len(addresses)

    print(f"  Fetching stats for {total} addresses from Alchemy...")

    # Use thread pool for parallel requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_addr = {
            executor.submit(
                fetch_address_stats_alchemy, addr, chain, api_key, base_url, prices
            ): addr
            for addr in addresses
        }

        for future in as_completed(future_to_addr):
            addr = future_to_addr[future]
            try:
                stats = future.result()
                all_stats[addr] = stats
            except Exception as e:
                all_stats[addr] = {}

            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"    Progress: {completed}/{total} addresses...")

    print(f"  Got stats for {len(all_stats)} addresses")
    return all_stats


def analyse_scores(
    chain: str,
    scores_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    top_count: int = 100,
    skip_alchemy: bool = False,
):
    """Main analysis function."""
    print("=" * 60)
    print("ChainRank Score Analyzer")
    print("=" * 60)
    print(f"\nChain: {chain}")
    print(f"Top EOAs to analyze: {top_count}")

    # Load identity data first (needed for filtering)
    print("\n[1/7] Loading ENS names...")
    ens_map = load_ens_names(raw_dir)
    ens_addresses = set(ens_map.keys())

    print("\n[2/7] Loading Farcaster usernames...")
    fc_map = load_farcaster_names(raw_dir)
    fc_addresses = set(fc_map.keys())

    print("\n[3/7] Loading World ID verifications...")
    wrld_set = load_worldcoin_addresses(raw_dir)

    # Step 2: Load top EOAs from scores (filtered by identity)
    print(f"\n[4/7] Loading top {top_count} EOAs from scores...")
    top_eoas = load_top_eoas_from_scores(
        scores_dir, chain, raw_dir, top_count, ens_addresses, fc_addresses, wrld_set
    )

    if not top_eoas:
        print("  Error: No EOAs found in scores file")
        return None

    addresses = [addr for addr, _, _, _ in top_eoas]
    scores = {addr: norm_score for addr, norm_score, _, _ in top_eoas}
    original_ranks = {addr: orig_rank for addr, _, orig_rank, _ in top_eoas}
    print(f"  Loaded {len(addresses)} EOA addresses")

    # Step 5: Fetch wallet stats from Alchemy
    print("\n[5/7] Fetching wallet stats from Alchemy...")
    if skip_alchemy:
        print("  Skipping Alchemy queries (--skip-alchemy flag)")
        wallet_stats = {}
    else:
        wallet_stats = fetch_wallet_stats_from_alchemy(addresses, chain)

    # Step 6: Build output
    print("\n[6/7] Building output...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"analysed_scores_{chain}.csv"

    with open(output_file, "w", newline="") as f:
        fieldnames = [
            "original_rank",
            "address",
            "trust_score",
            "net_worth",
            "wallet_volume_6m",
            "tx_count",
            "last_active",
            "first_active",
            "ens",
            "farcaster",
            "world_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for addr in addresses:
            stats = wallet_stats.get(addr, {})
            net_worth = stats.get("net_worth_usd", 0)
            wallet_vol = stats.get("wallet_volume_usd", 0)
            row = {
                "original_rank": original_ranks.get(addr, ""),
                "address": addr,
                "trust_score": f"{scores.get(addr, 0):.6f}",
                "net_worth": format_currency(net_worth) if net_worth else "",
                "wallet_volume_6m": format_currency(wallet_vol) if wallet_vol else "",
                "tx_count": stats.get("tx_count", ""),
                "last_active": format_timestamp(stats.get("last_active")),
                "first_active": format_timestamp(stats.get("first_active")),
                "ens": ens_map.get(addr, ""),
                "farcaster": fc_map.get(addr, ""),
                "world_id": "true" if addr in wrld_set else "false",
            }
            writer.writerow(row)

    print(f"\n✓ Output saved to: {output_file}")

    # Print summary statistics
    ens_count = sum(1 for addr in addresses if addr in ens_map)
    fc_count = sum(1 for addr in addresses if addr in fc_map)
    wrld_count = sum(1 for addr in addresses if addr in wrld_set)
    alchemy_count = sum(1 for addr in addresses if wallet_stats.get(addr))

    print(f"\nSummary:")
    print(f"  Total EOAs analyzed: {len(addresses)}")
    print(f"  With Alchemy stats: {alchemy_count}")
    print(f"  With ENS: {ens_count}")
    print(f"  With Farcaster: {fc_count}")
    print(f"  With World ID: {wrld_count}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Analyze top EOA addresses from ChainRank scores"
    )
    parser.add_argument(
        "--chain",
        type=str,
        required=True,
        help="Chain to analyze (e.g., ethereum, base)",
    )
    parser.add_argument(
        "--scores-dir",
        type=str,
        default="scores",
        help="Directory containing score files (default: scores)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="raw",
        help="Directory containing raw data files (ENS, Farcaster, World ID)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/output",
        help="Directory to save output",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top EOAs to analyze (default: 100)",
    )
    parser.add_argument(
        "--skip-alchemy",
        action="store_true",
        help="Skip Alchemy API queries (only use local data)",
    )

    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not scores_dir.exists():
        print(f"Error: Scores directory not found: {scores_dir}")
        return 1

    if not raw_dir.exists():
        print(f"Error: Raw data directory not found: {raw_dir}")
        return 1

    result = analyse_scores(
        chain=args.chain.lower(),
        scores_dir=scores_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        top_count=args.top,
        skip_alchemy=args.skip_alchemy,
    )

    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
