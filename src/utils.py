"""
General-purpose utilities for WasteWise AI.

These helpers intentionally contain no factory-specific assumptions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    """
    Normalize arbitrary text for comparisons.

    Examples:
        "Production Line" -> "production line"
        "Waste_Qty"       -> "waste qty"
    """
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_column_name(value: Any) -> str:
    """
    Normalize a column name into a comparison-friendly representation.

    Examples:
        "Production Line" -> "production_line"
        "PROD-QTY"        -> "prod_qty"
    """
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def safe_numeric(
    series: pd.Series,
) -> tuple[pd.Series, int]:
    """
    Convert a pandas Series to numeric values safely.

    Commas and surrounding whitespace are tolerated.

    Returns:
        converted_series, invalid_value_count
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
    )

    converted = pd.to_numeric(cleaned, errors="coerce")

    original_non_null = series.notna()
    invalid_count = int(
        (original_non_null & converted.isna()).sum()
    )

    return converted, invalid_count


def safe_datetime(
    series: pd.Series,
) -> tuple[pd.Series, int]:
    """
    Convert a pandas Series to datetime safely.

    Returns:
        converted_series, invalid_value_count
    """
    converted = pd.to_datetime(
        series,
        errors="coerce",
    )

    original_non_null = series.notna()
    invalid_count = int(
        (original_non_null & converted.isna()).sum()
    )

    return converted, invalid_count


def get_file_extension(filename: str) -> str:
    """Return a lowercase file extension without the leading dot."""
    return Path(filename).suffix.lower().lstrip(".")


def is_supported_data_file(filename: str) -> bool:
    """Return whether the filename is a supported factory-data file."""
    return get_file_extension(filename) in {"csv", "xlsx", "xls"}


def clean_string_series(series: pd.Series) -> pd.Series:
    """Clean string-like dataframe values without changing nulls."""
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA})
    )
