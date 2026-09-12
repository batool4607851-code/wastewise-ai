from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .utils import (
    clean_string_series,
    normalize_column_name,
    safe_datetime,
    safe_numeric,
)


CANONICAL_FIELDS = [
    "date",
    "product",
    "batch",
    "production_line",
    "shift",
    "production_quantity",
    "event_type",
    "event_reason",
    "affected_quantity",
    "loss_quantity",
    "rework_quantity",
    "recovered_quantity",
    "final_loss_quantity",
    "disposition",
    "unit_cost",
    "rework_cost",
    "loss_cost",
    "quality_parameter",
    "quality_value",
    "quality_status",
]


REQUIRED_CANONICAL_FIELDS = [
    "date",
    "product",
    "batch",
    "production_line",
    "shift",
    "production_quantity",
    "event_type",
    "event_reason",
    "affected_quantity",
    "unit_cost",
]


FIELD_ALIASES: dict[str, list[str]] = {
    "date": [
        "date",
        "datetime",
        "timestamp",
        "event_date",
        "production_date",
        "record_date",
    ],
    "product": [
        "product",
        "product_name",
        "sku",
        "item",
        "item_name",
    ],
    "batch": [
        "batch",
        "batch_id",
        "batch_number",
        "lot",
        "lot_number",
        "production_batch",
    ],
    "production_line": [
        "production_line",
        "productionline",
        "line",
        "line_name",
        "line_id",
        "machine_line",
    ],
    "shift": [
        "shift",
        "shift_name",
        "work_shift",
    ],
    "production_quantity": [
        "production_quantity",
        "production_qty",
        "prod_qty",
        "produced_quantity",
        "produced_qty",
        "output_quantity",
        "output_qty",
        "production_volume",
    ],
    "event_type": [
        "event_type",
        "event",
        "event_category",
        "loss_type",
        "activity_type",
        "record_type",
    ],
    "event_reason": [
        "event_reason",
        "reason",
        "waste_reason",
        "loss_reason",
        "cause",
        "reason_code",
        "waste_cause",
    ],
    "affected_quantity": [
        "affected_quantity",
        "affected_qty",
        "waste_quantity",
        "waste_qty",
        "quantity_affected",
        "quantity",
    ],
    "loss_quantity": [
        "loss_quantity",
        "loss_qty",
        "material_loss",
        "material_loss_quantity",
        "waste_loss",
    ],
    "rework_quantity": [
        "rework_quantity",
        "rework_qty",
        "quantity_reworked",
        "rework_volume",
    ],
    "recovered_quantity": [
        "recovered_quantity",
        "recovered_qty",
        "recovery_quantity",
        "recovery_qty",
        "quantity_recovered",
    ],
    "final_loss_quantity": [
        "final_loss_quantity",
        "final_loss_qty",
        "final_loss",
        "net_loss_quantity",
        "actual_loss_quantity",
    ],
    "disposition": [
        "disposition",
        "final_disposition",
        "material_disposition",
        "outcome",
    ],
    "unit_cost": [
        "unit_cost",
        "cost_per_unit",
        "unit_price",
        "cost_unit",
    ],
    "rework_cost": [
        "rework_cost",
        "cost_of_rework",
        "rework_expense",
    ],
    "loss_cost": [
        "loss_cost",
        "waste_cost",
        "cost_of_loss",
        "loss_expense",
    ],
    "quality_parameter": [
        "quality_parameter",
        "quality_metric",
        "parameter",
        "quality_measure",
    ],
    "quality_value": [
        "quality_value",
        "quality_measurement",
        "measured_value",
        "parameter_value",
    ],
    "quality_status": [
        "quality_status",
        "quality_result",
        "qc_status",
        "inspection_status",
        "status",
    ],
}


NUMERIC_FIELDS = {
    "production_quantity",
    "affected_quantity",
    "loss_quantity",
    "rework_quantity",
    "recovered_quantity",
    "final_loss_quantity",
    "unit_cost",
    "rework_cost",
    "loss_cost",
    "quality_value",
}


STRING_FIELDS = {
    "product",
    "batch",
    "production_line",
    "shift",
    "event_type",
    "event_reason",
    "disposition",
    "quality_parameter",
    "quality_status",
}


@dataclass
class MappingResult:
    canonical_dataframe: pd.DataFrame | None = None
    mappings: dict[str, str] = field(default_factory=dict)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    unmapped_source_columns: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    conversion_warnings: list[str] = field(default_factory=list)


def _normalise_aliases(canonical_field: str) -> set[str]:
    return {
        normalize_column_name(alias)
        for alias in FIELD_ALIASES.get(
            canonical_field,
            [],
        )
    }


def _score_column_match(
    source_column: str,
    normalized_source: str,
    canonical_field: str,
) -> float:
    canonical_normalized = normalize_column_name(
        canonical_field
    )

    aliases = _normalise_aliases(
        canonical_field
    )

    if normalized_source == canonical_normalized:
        return 100.0

    if normalized_source in aliases:
        return 95.0

    source_tokens = set(
        normalized_source.split("_")
    )

    canonical_tokens = set(
        canonical_normalized.split("_")
    )

    if (
        canonical_tokens
        and canonical_tokens.issubset(source_tokens)
    ):
        return 80.0

    for alias in aliases:
        alias_tokens = set(alias.split("_"))

        if (
            alias_tokens
            and alias_tokens.issubset(source_tokens)
        ):
            return 70.0

    if canonical_normalized in normalized_source:
        return 50.0

    for alias in aliases:
        if alias and alias in normalized_source:
            return 40.0

    return 0.0


def map_columns(
    dataframe: pd.DataFrame,
) -> MappingResult:
    """
    Dynamically map source columns to the canonical schema.

    If multiple source columns are equally strong candidates
    for the same canonical field, the field is marked ambiguous.
    The mapper never silently guesses between equivalent columns.
    """
    source_columns = list(
        dataframe.columns
    )

    normalized_sources = {
        column: normalize_column_name(column)
        for column in source_columns
    }

    mappings: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    warnings: list[str] = []

    used_source_columns: set[str] = set()

    for canonical_field in CANONICAL_FIELDS:

        candidates: list[tuple[str, float]] = []

        for source_column in source_columns:
            if source_column in used_source_columns:
                continue

            normalized_source = (
                normalized_sources[source_column]
            )

            score = _score_column_match(
                source_column,
                normalized_source,
                canonical_field,
            )

            if score > 0:
                candidates.append(
                    (source_column, score)
                )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        best_score = candidates[0][1]

        best_candidates = [
            column
            for column, score in candidates
            if score == best_score
        ]

        if len(best_candidates) > 1:
            ambiguous[canonical_field] = (
                best_candidates
            )

            warnings.append(
                f"Ambiguous mapping for "
                f"'{canonical_field}': "
                f"{', '.join(best_candidates)}. "
                "Confirmation is required."
            )

            continue

        selected_column = best_candidates[0]

        mappings[canonical_field] = (
            selected_column
        )

        used_source_columns.add(
            selected_column
        )

    unmapped_source_columns = [
        column
        for column in source_columns
        if column not in used_source_columns
    ]

    missing_fields = [
        field_name
        for field_name in REQUIRED_CANONICAL_FIELDS
        if field_name not in mappings
    ]

    if missing_fields:
        warnings.append(
            "Missing required canonical fields: "
            + ", ".join(missing_fields)
        )

    return MappingResult(
        mappings=mappings,
        ambiguous=ambiguous,
        unmapped_source_columns=(
            unmapped_source_columns
        ),
        missing_fields=missing_fields,
        warnings=warnings,
    )


def _normalization_warnings(
    value: Any,
    field_name: str,
) -> list[str]:
    """
    Convert warning metadata returned by utility
    functions into the MappingResult warning format.

    The current utility functions return a warning
    count rather than a list of warning strings.
    """
    if value is None:
        return []

    if isinstance(value, (int, float)):
        if value > 0:
            return [
                f"{int(value)} value(s) could not be "
                f"normalized for '{field_name}'."
            ]

        return []

    if isinstance(value, str):
        return [value]

    try:
        return list(value)
    except TypeError:
        return []


def normalize_canonical_dataframe(
    dataframe: pd.DataFrame,
    mappings: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create a canonical dataframe using confirmed mappings.

    Missing canonical fields are left as pd.NA.
    They are never replaced with zero.
    """
    warnings: list[str] = []

    canonical = pd.DataFrame(
        index=dataframe.index
    )

    for field_name in CANONICAL_FIELDS:
        source_column = mappings.get(
            field_name
        )

        if source_column is None:
            canonical[field_name] = pd.NA
            continue

        if source_column not in dataframe.columns:
            canonical[field_name] = pd.NA

            warnings.append(
                f"Mapped source column "
                f"'{source_column}' for "
                f"'{field_name}' is no longer "
                "available."
            )

            continue

        series = dataframe[
            source_column
        ]

        if field_name == "date":
            normalized, warning_info = (
                safe_datetime(series)
            )

            canonical[field_name] = (
                normalized
            )

            warnings.extend(
                _normalization_warnings(
                    warning_info,
                    field_name,
                )
            )

        elif field_name in NUMERIC_FIELDS:
            normalized, warning_info = (
                safe_numeric(series)
            )

            canonical[field_name] = (
                normalized
            )

            warnings.extend(
                _normalization_warnings(
                    warning_info,
                    field_name,
                )
            )

        elif field_name in STRING_FIELDS:
            canonical[field_name] = (
                clean_string_series(series)
            )

        else:
            canonical[field_name] = series

    return canonical, warnings


def create_mapping_result(
    dataframe: pd.DataFrame,
) -> MappingResult:
    """
    Map source columns and normalize the dataframe.

    Normalization occurs only for fields with an
    unambiguous mapping.
    """
    result = map_columns(
        dataframe
    )

    if result.errors:
        return result

    canonical_dataframe, conversion_warnings = (
        normalize_canonical_dataframe(
            dataframe,
            result.mappings,
        )
    )

    result.canonical_dataframe = (
        canonical_dataframe
    )

    result.conversion_warnings = (
        conversion_warnings
    )

    result.warnings.extend(
        conversion_warnings
    )

    return result


def describe_mapping(
    result: MappingResult,
) -> dict[str, Any]:
    """
    Return a UI-friendly summary of a mapping result.
    """
    return {
        "mappings": result.mappings,
        "ambiguous": result.ambiguous,
        "missing_fields": (
            result.missing_fields
        ),
        "unmapped_source_columns": (
            result.unmapped_source_columns
        ),
        "warnings": result.warnings,
        "errors": result.errors,
        "conversion_warnings": (
            result.conversion_warnings
        ),
    }
