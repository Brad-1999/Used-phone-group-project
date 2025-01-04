import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import polars as pl
from loguru import logger
from tqdm import tqdm


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


def safe_get(obj: SimpleNamespace, *attrs: str) -> Optional[Any]:
    """Safely get nested attributes from an object."""
    try:
        current = obj
        for attr in attrs:
            current = getattr(current, attr)
        return current
    except AttributeError:
        return None


def get_ad_info(obj: NestedNamespace) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Extract ad and ad_params from nested object."""
    ad = (
        safe_get(
            obj,
            "content",
            "props",
            "pageProps",
            "initialState",
            "adView",
            "adInfo",
            "ad",
        )
        or SimpleNamespace()
    )

    ad_params = (
        safe_get(
            obj,
            "content",
            "props",
            "pageProps",
            "initialState",
            "adView",
            "adInfo",
            "ad_params",
        )
        or SimpleNamespace()
    )

    return ad, ad_params


def extract_description(ad: SimpleNamespace) -> Dict[str, Any]:
    """Extract description fields from ad object."""
    return {
        "ad_id": getattr(ad, "ad_id", None),
        "list_id": getattr(ad, "list_id", None),
        "subject": getattr(ad, "subject", None),
        "body": getattr(ad, "body", None),
    }


def format_timestamp(timestamp: Optional[int]) -> Optional[str]:
    """Convert millisecond timestamp to YYYY-MM-DD format."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp / 1000.0).strftime(r"%Y-%m-%d")


def extract_info(
    ad: SimpleNamespace, ad_params: SimpleNamespace, obj: NestedNamespace
) -> Dict[str, Any]:
    """Extract detailed info fields from ad and ad_params objects."""

    def get_param_value(param_name: str) -> Optional[str]:
        try:
            return getattr(ad_params, param_name).value
        except AttributeError:
            return None

    return {
        "ad_id": getattr(ad, "ad_id", None),
        "list_id": getattr(ad, "list_id", None),
        "list_time": format_timestamp(getattr(ad, "list_time", None)),
        "account_name": getattr(ad, "account_name", None),
        "phone": getattr(ad, "phone", None),
        "company_ad": getattr(ad, "company_ad", None),
        "price": getattr(ad, "price", None),
        "account_id": getattr(ad, "account_id", None),
        "longitude": getattr(ad, "longitude", None),
        "latitude": getattr(ad, "latitude", None),
        "full_name": getattr(ad, "full_name", None),
        "sold_ads": getattr(ad, "sold_ads", None),
        "total_rating": getattr(ad, "total_rating", None),
        "total_rating_for_seller": getattr(ad, "total_rating_for_seller", None),
        "average_rating": getattr(ad, "average_rating", None),
        "average_rating_for_seller": getattr(ad, "average_rating_for_seller", None),
        "account_oid": getattr(ad, "account_oid", None),
        "area_name": getattr(ad, "area_name", None),
        "region_name": getattr(ad, "region_name", None),
        "number_of_images": getattr(ad, "number_of_images", None),
        "ward_name": getattr(ad, "ward_name", None),
        "address": get_param_value("address"),
        "elt_condition": get_param_value("elt_condition"),
        "elt_origin": get_param_value("elt_origin"),
        "elt_warranty": get_param_value("elt_warranty"),
        "mobile_brand": get_param_value("mobile_brand"),
        "mobile_capacity": get_param_value("mobile_capacity"),
        "mobile_color": get_param_value("mobile_color"),
        "mobile_model": get_param_value("mobile_model"),
        "usage_information": get_param_value("usage_information"),
        "url": getattr(obj, "url", None),
    }


def extract_csv(input_file: str, output_dir: str = "data") -> dict[str, pl.DataFrame]:
    """
    Extract data from JSONL file and save as CSV files.

    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save output CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    info_records: List[Dict[str, Any]] = []
    description_records: List[Dict[str, Any]] = []

    try:
        with open(input_file) as file:
            for line in tqdm(file, mininterval=1):
                obj = NestedNamespace(json.loads(line))
                ad, ad_params = get_ad_info(obj)

                description_records.append(extract_description(ad))
                info_records.append(extract_info(ad, ad_params, obj))

        description = pl.DataFrame(description_records)
        info = pl.DataFrame(info_records)
        description.write_csv(output_path / "description.csv")
        info.write_csv(output_path / "info.csv")
        logger.info(f"Successfully extracted {len(info_records)} records to CSV files")

        return {
            "description": description,
            "info": info,
        }

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise


if __name__ == "__main__":
    extract_csv("data/2024-10-24.jsonl", "data")
