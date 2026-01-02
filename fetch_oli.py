#!/usr/bin/env python3
"""
Open Labels Initiative (OLI) Data Fetcher

Downloads blockchain address label data from the Open Labels Initiative.
Fetches only essential fields: address, contract_name, is_eoa, is_contract,
is_proxy, is_safe_contract, deployer_address.

Usage:
    python fetch_oli.py

Requirements:
    pip install pandas pyarrow requests
"""

import os

import pandas as pd
import requests

OUTPUT_FILE = "raw/oli_attestations.csv"

# Direct URL to the decoded parquet export (from growthepie API)
PARQUET_URL = "https://api.growthepie.com/v1/oli/labels_decoded.parquet"

# Only fetch data for Ethereum mainnet
ETHEREUM_CHAIN_ID = "eip155:1"

REQUIRED_TAGS = [
    "contract_name",
    "is_eoa",
    "is_contract",
    "is_proxy",
    "is_safe_contract",
    "deployer_address",
]


def _parse_bool(value):
    """Parse boolean value from various formats."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def fetch_oli_data():
    """Fetch OLI data and extract only required fields."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    temp_parquet = "raw/temp_oli_export.parquet"

    print(f"Downloading OLI parquet export from {PARQUET_URL}...")
    response = requests.get(PARQUET_URL, stream=True)
    response.raise_for_status()

    with open(temp_parquet, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {temp_parquet}")

    # Read and process the data
    print("Processing data...")
    df = pd.read_parquet(temp_parquet)

    print(f"Total records in export: {len(df)}")
    print(f"Available columns: {list(df.columns)}")

    # Filter to only Ethereum mainnet (eip155:1)
    if "chain_id" in df.columns:
        df = df[df["chain_id"] == ETHEREUM_CHAIN_ID]
        print(f"Filtered to Ethereum mainnet ({ETHEREUM_CHAIN_ID}): {len(df)} records")
    elif "chainId" in df.columns:
        df = df[df["chainId"] == ETHEREUM_CHAIN_ID]
        print(f"Filtered to Ethereum mainnet ({ETHEREUM_CHAIN_ID}): {len(df)} records")

    # Extract only the fields we need
    extracted_data = []

    # Group by address and chain_id to consolidate tags
    if "address" in df.columns and "chain_id" in df.columns:
        grouped = df.groupby(["address", "chain_id"])

        for (address, chain_id), group in grouped:
            record = {
                "address": address,
                "chain_id": chain_id,
                "contract_name": None,
                "is_eoa": None,
                "is_contract": None,
                "is_proxy": None,
                "is_safe_contract": None,
                "deployer_address": None,
            }

            # Extract tags from the group
            for _, row in group.iterrows():
                tag_id = row.get("tag_id") or row.get("tagId")
                tag_value = (
                    row.get("tag_value") or row.get("tagValue") or row.get("value")
                )

                if tag_id == "contract_name":
                    record["contract_name"] = tag_value
                elif tag_id == "is_eoa":
                    record["is_eoa"] = _parse_bool(tag_value)
                elif tag_id == "is_contract":
                    record["is_contract"] = _parse_bool(tag_value)
                elif tag_id == "is_proxy":
                    record["is_proxy"] = _parse_bool(tag_value)
                elif tag_id == "is_safe_contract":
                    record["is_safe_contract"] = _parse_bool(tag_value)
                elif tag_id == "deployer_address":
                    record["deployer_address"] = tag_value

            # Only include records that have at least one of our required fields
            if any(record[f] is not None for f in REQUIRED_TAGS):
                extracted_data.append(record)
    else:
        # Alternative parsing if structure is different
        for _, row in df.iterrows():
            record = {
                "address": row.get("address") or row.get("recipient"),
                "chain_id": row.get("chain_id") or row.get("chainId"),
                "contract_name": row.get("contract_name"),
                "is_eoa": _parse_bool(row.get("is_eoa")),
                "is_contract": _parse_bool(row.get("is_contract")),
                "is_proxy": _parse_bool(row.get("is_proxy")),
                "is_safe_contract": _parse_bool(row.get("is_safe_contract")),
                "deployer_address": row.get("deployer_address"),
            }

            if record["address"] and any(record[f] is not None for f in REQUIRED_TAGS):
                extracted_data.append(record)

    print(f"Extracted {len(extracted_data)} records with required fields")

    # Save to CSV
    result_df = pd.DataFrame(extracted_data)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved extracted data to {OUTPUT_FILE}")

    # Clean up temp file
    if os.path.exists(temp_parquet):
        os.remove(temp_parquet)
        print("Cleaned up temporary file")

    return OUTPUT_FILE


def main():
    fetch_oli_data()


if __name__ == "__main__":
    main()
