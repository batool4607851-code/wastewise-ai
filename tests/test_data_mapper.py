import pandas as pd

from src.data_mapper import (
    CANONICAL_FIELDS,
    create_mapping_result,
    map_columns,
)


def test_exact_and_alias_mapping():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product Name": ["Sauce A"],
            "Prod Qty": ["1,000"],
            "Production Line": ["Line 1"],
            "Waste Qty": ["25"],
            "Unit Cost": ["450"],
        }
    )

    result = create_mapping_result(
        dataframe
    )

    assert result.mappings["date"] == "Date"
    assert result.mappings["product"] == "Product Name"
    assert result.mappings["production_quantity"] == "Prod Qty"
    assert result.mappings["production_line"] == "Production Line"
    assert result.mappings["loss_quantity"] == "Waste Qty"
    assert result.mappings["unit_cost"] == "Unit Cost"


def test_canonical_schema_is_present():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product": ["A"],
            "Production Qty": [100],
        }
    )

    result = create_mapping_result(
        dataframe
    )

    assert result.canonical_dataframe is not None

    for field in CANONICAL_FIELDS:
        assert field in result.canonical_dataframe.columns


def test_numeric_values_are_normalized():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product": ["A"],
            "Production Qty": ["1,250"],
            "Waste Qty": ["25"],
        }
    )

    result = create_mapping_result(
        dataframe
    )

    canonical = result.canonical_dataframe

    assert canonical is not None
    assert canonical["production_quantity"].iloc[0] == 1250
    assert canonical["loss_quantity"].iloc[0] == 25


def test_invalid_numeric_values_generate_warning():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product": ["A"],
            "Production Qty": ["not-a-number"],
        }
    )

    result = create_mapping_result(
        dataframe
    )

    assert any(
        "production_quantity" in warning
        for warning in result.warnings
    )

    assert pd.isna(
        result.canonical_dataframe[
            "production_quantity"
        ].iloc[0]
    )


def test_missing_required_field_is_reported():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product": ["A"],
        }
    )

    result = create_mapping_result(
        dataframe
    )

    assert "production_quantity" in result.missing_fields

    assert any(
        "production_quantity" in warning
        for warning in result.warnings
    )


def test_missing_fields_are_not_fabricated_as_zero():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product": ["A"],
            "Production Qty": [100],
        }
    )

    result = create_mapping_result(
        dataframe
    )

    canonical = result.canonical_dataframe

    assert canonical is not None
    assert pd.isna(
        canonical["loss_quantity"].iloc[0]
    )
    assert pd.isna(
        canonical["rework_quantity"].iloc[0]
    )
    assert pd.isna(
        canonical["recovered_quantity"].iloc[0]
    )


def test_ambiguous_mapping_is_not_silently_selected():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product": ["A"],
            "Production": [100],
            "Qty": [25],
        }
    )

    result = map_columns(
        dataframe
    )

    # "Qty" is intentionally not an alias for a single
    # canonical quantity field, so it should not be
    # silently interpreted as loss/rework/production.
    assert "Qty" in result.unmapped_source_columns


def test_date_is_normalized():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-15"],
            "Product": ["A"],
            "Production Qty": [100],
        }
    )

    result = create_mapping_result(
        dataframe
    )

    canonical = result.canonical_dataframe

    assert canonical is not None
    assert pd.api.types.is_datetime64_any_dtype(
        canonical["date"]
    )


def test_duplicate_source_mapping_is_not_reused():
    dataframe = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Product": ["A"],
            "Production Qty": [100],
            "Production Quantity": [110],
        }
    )

    result = map_columns(
        dataframe
    )

    assert (
        "production_quantity"
        in result.ambiguous
    )
