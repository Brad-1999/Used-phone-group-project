from typing import Dict

import polars as pl
import unidecode

USD2VND = 25418  # The exchange rate is 1 USD (US Dollars) = 25,418 VND (Vietnam Thousand Dong) in 11-23-25

CONDITION_MAPPING = {
    "Đã sử dụng (chưa sửa chữa)": "used",
    "Đã sử dụng (qua sửa chữa)": "refurbished",
    "Mới": "new",
}

CAPACITY_MAPPING = {
    "< 8GB": "less_than_8",
    "8 GB": "8",
    "16 GB": "16",
    "32 GB": "32",
    "64 GB": "64",
    "128 GB": "128",
    "256 GB": "256",
    "512 GB": "512",
    "1 TB": "1024",
    "> 2 TB": "more_than_2048",
}

WARRANTY_MAPPING = {
    "1 tháng": "1",
    "2 tháng": "2",
    "3 tháng": "3",
    "4-6 tháng": "4_to_6",
    "7-12 tháng": "7_to_12",
    ">12 tháng": "more_than_12",
    "Còn bảo hành": "active",
    "Bảo hành hãng": "manufacturer",
    "Hết bảo hành": "expired",
}

COLOR_MAPPING = {
    "Đỏ": "red",
    "Bạc": "silver",
    "Vàng": "gold",
    "Vàng hồng": "rose_gold",
    "Xám": "gray",
    "Xanh dương": "blue",
    "Đen": "black",
    "Đen bóng - Jet black": "black",
    "Trắng": "white",
    "Hồng": "pink",
    "Xanh lá": "green",
    "Cam": "orange",
    "Tím": "purple",
    "Màu khác": "other",
}

ORIGIN_MAPPING = {
    "Đức": "germany",
    "Thái Lan": "thailand",
    "Hàn Quốc": "south_korea",
    "Đang cập nhật": "unknown",
    "Việt Nam": "vietnam",
    "Mỹ": "usa",
    "Đài Loan": "taiwan",
    "Nước khác": "other",
    "Ấn Độ": "india",
    "Nhật Bản": "japan",
    "Trung Quốc": "china",
}

BRAND_MAPPING = {
    "Q Mobile": "QMobile",
    "Nokia thông minh": "Nokia_Smart",
    "Nokia phổ thông": "Nokia_Feature",
}

CITY_MAPPING = {
    "Tp Hồ Chí Minh": "ho_chi_minh_city",
    "Hà Nội": "hanoi",
    "Đà Nẵng": "da_nang",
    "Hải Phòng": "hai_phong",
    "Cần Thơ": "can_tho",
}

COMPOUND_MAPPING = {
    "Bà Rịa - Vũng Tàu": "ba_ria_vung_tau",
    "Thừa Thiên Huế": "thua_thien_hue",
}


def standardize_text(text: str) -> str:
    """
    Transliterates and standardizes a given text.

    This function removes diacritics, converts the text to lowercase,
    and replaces spaces and hyphens with underscores.

    Args:
        text: The text to standardize.

    Returns:
        The standardized text.
    """
    text = unidecode.unidecode(text)
    return text.lower().replace(" ", "_").replace("-", "_")


def standardize_location_name(
    name: str, postfix_mapping: Dict[str, str], is_compound: bool = False
) -> str:
    """Standardizes a location name by handling postfixes and transliterating."""

    # Handle the special case of unwanted character \x08
    name = name.replace("\x08", "")

    parts = name.split()

    # For "Thị trấn" postfix handling
    if is_compound and parts[0] == "Thị" and parts[1] == "trấn":
        postfix = "Thị trấn"
        location = " ".join(parts[2:])
    else:
        postfix = " ".join(parts[:2]) if parts[0] in ["Thành", "Thị"] else parts[0]
        location = (
            " ".join(parts[2:]) if parts[0] in ["Thành", "Thị"] else " ".join(parts[1:])
        )

    # Get the translated postfix, use original if not found
    new_postfix = postfix_mapping.get(postfix, postfix)

    # Transliterate and format the location name
    location = standardize_text(location)

    return f"{location}_{new_postfix}"


def standardize_area_name(name: str) -> str:
    """Standardizes the area name using a predefined mapping."""
    postfix_mapping = {
        "Thành phố": "city",  # Removed "of" for brevity and consistency
        "Thị xã": "town",  # Smaller administrative division than a city
        "Quận": "district",  # Urban district
        "Huyện": "rural_district",  # Rural district
    }

    return standardize_location_name(name, postfix_mapping)


def standardize_ward_name(name: str) -> str:
    """Standardizes the ward name using a predefined mapping."""
    postfix_mapping = {
        "Phường": "ward",  # Urban ward
        "Xã": "commune",  # Rural commune
        "Thị trấn": "township",  # Township
    }

    return standardize_location_name(name, postfix_mapping, is_compound=True)


def standardize_region_name(name: str) -> str:
    """Standardizes a region name by mapping special names or formatting."""
    if name in CITY_MAPPING:
        return CITY_MAPPING[name]
    if name in COMPOUND_MAPPING:
        return COMPOUND_MAPPING[name]
    return standardize_text(name)


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.with_columns(
            pl.col("list_time").str.to_date(),
            pl.col("company_ad").fill_null(False).alias("is_company"),
        )
        .filter(
            (pl.col("mobile_brand") != "Hãng khác")
            & (pl.col("mobile_model") != "Dòng khác")
            & (pl.col("mobile_capacity") != "8")
            & (pl.col("price") < 60_000_000)
        )
        .drop(
            "full_name",
            "company_ad",
            "phone",
            "address",
            "usage_information",
            "url",
        )
        .rename(
            {
                "elt_condition": "condition",
                "elt_origin": "origin",
                "elt_warranty": "warranty",
                "mobile_brand": "brand",
                "mobile_capacity": "capacity",
                "mobile_color": "color",
                "mobile_model": "model",
            }
        )
    )

    df = df.with_columns(
        pl.col("condition").replace(CONDITION_MAPPING),
        pl.col("sold_ads").fill_null(0),
        pl.col("total_rating").fill_null(0),
        pl.col("total_rating_for_seller").fill_null(0),
        pl.col("average_rating").fill_null(-1),
        pl.col("average_rating_for_seller").fill_null(-1),
        pl.col("capacity").replace(CAPACITY_MAPPING),
        pl.col("warranty").replace(WARRANTY_MAPPING),
        pl.col("color").replace(COLOR_MAPPING),
        pl.col("origin").replace(ORIGIN_MAPPING),
        pl.col("brand").replace(BRAND_MAPPING),
        (pl.col("price") / USD2VND).round(decimals=2).alias("price"),
    )

    df = df.with_columns(
        pl.col("region_name").map_elements(
            standardize_region_name, return_dtype=pl.String
        ),
        pl.col("area_name").map_elements(standardize_area_name, return_dtype=pl.String),
        pl.col("ward_name").map_elements(standardize_ward_name, return_dtype=pl.String),
    )
    return df


if __name__ == "__main__":
    df = pl.read_csv("data/info.csv")
    cleaned_df = preprocess(df)
    cleaned_df.write_csv("data/cleaned_info.csv")
