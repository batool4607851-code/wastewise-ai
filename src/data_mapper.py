from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

NUMERIC_CANONICAL_FIELDS = {
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

STRING_CANONICAL_FIELDS = {
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


ALIASES: Dict[str, List[str]] = {
    "date": [
        "date",
        "datetime",
        "timestamp",
        "event date",
        "production date",
        "waste date",
    ],
    "product": [
        "product",
        "product name",
        "item",
        "item name",
        "sku",
        "product sku",
    ],
    "batch": [
        "batch",
        "batch id",
        "batch number",
        "lot",
        "lot number",
        "production batch",
    ],
    "production_line": [
        "production line",
        "production line id",
        "line",
        "line id",
        "line number",
    ],
    "shift": [
        "shift",
        "production shift",
        "work shift",
    ],
    "production_quantity": [
        "production quantity",
        "production qty",
        "prod quantity",
        "prod qty",
        "produced quantity",
        "produced qty",
        "output quantity",
        "output qty",
        "production volume",
    ],
    "event_type": [
        "event type",
        "type",
        "waste type",
        "loss type",
        "record type",
    ],
    "event_reason": [
        "event reason",
        "reason",
        "waste reason",
        "loss reason",
        "cause",
        "reason code",
    ],
    "affected_quantity": [
        "affected quantity",
        "affected qty",
        "quantity affected",
        "qty affected",
    ],
    "loss_quantity": [
        "loss quantity",
        "loss qty",
        "loss amount",
        "loss volume",
        "final loss quantity",
        "final loss qty",
        "waste quantity",
        "waste qty",
        "waste amount",
        "waste volume",
        "quantity wasted",
        "qty wasted",
        "wasted quantity",
        "wasted qty",
        "scrap quantity",
        "scrap qty",
    ],
    "rework_quantity": [
        "rework quantity",
        "rework qty",
        "rework amount",
        "reworked quantity",
        "reworked qty",
    ],
    "recovered_quantity": [
        "recovered quantity",
        "recovered qty",
        "recovery quantity",
        "recovery qty",
        "salvaged quantity",
        "salvaged qty",
    ],
    "final_loss_quantity": [
        "final loss quantity",
        "final loss qty",
        "net loss quantity",
        "net loss qty",
        "actual loss quantity",
        "actual loss qty",
    ],
    "disposition": [
        "disposition",
        "disposal",
        "final disposition",
        "action",
        "resolution",
    ],
    "unit_cost": [
        "unit cost",
        "cost per unit",
        "unit price",
        "cost/unit",
        "price per unit",
    ],
    "rework_cost": [
        "rework cost",
        "cost of rework",
        "rework expense",
    ],
    "loss_cost": [
        "loss cost",
        "waste cost",
        "cost of loss",
        "cost of waste",
        "waste expense",
    ],
    "quality_parameter": [
        "quality parameter",
        "quality metric",
        "parameter",
        "metric",
        "test parameter",
    ],
    "quality_value": [
        "quality value",
        "measured value",
        "measurement",
        "test value",
        "result value",
    ],
    "quality_status": [
        "quality status",
        "status",
        "qc status",
        "quality result",
        "inspection status",
    ],
}


@dataclass
class MappingCandidate:
    canonical_field: str
    source_column: str
    score: int
    reason: str


@dataclass
class MappingResult:
    canonical_dataframe: Optional[pd.DataFrame] = None
    mappings: Dict[str, str] = field(default_factory=dict)
    unmapped_source_columns: List[str] = field(default_factory=list)
    ambiguous: Dict[str, List[str]] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    conversion_warnings: List[str] = field(default_factory=list)


def _normalization_warnings(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, int):
        if value == 0:
            return []
        return [f"{value} values could not be converted"]

    if isinstance(value, str):
        return [value] if value else []

    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return [str(value)]


def _candidate_score(
    source_column: str,
    canonical_field: str,
) -> Tuple[int, str]:
    source_normalized = normalize_column_name(source_column)
    canonical_normalized = normalize_column_name(canonical_field)

    aliases = {
        normalize_column_name(alias)
        for alias in ALIASES.get(canonical_field, [])
    }

    # Exact canonical names and known aliases intentionally receive
    # the same score. This is important because a source file containing
    # both "Production Quantity" and "Production Qty" must be flagged
    # as ambiguous rather than silently selecting one.
    if source_normalized == canonical_normalized:
        return 95, "canonical name"

    if source_normalized in aliases:
        return 95, "known alias"

    source_tokens = set(source_normalized.split())
    canonical_tokens = set(canonical_normalized.split())

    if canonical_tokens and canonical_tokens.issubset(source_tokens):
        return 70, "canonical tokens contained in source name"

    return 0, ""


def _build_candidates(
    dataframe: pd.DataFrame,
) -> Dict[str, List[MappingCandidate]]:
    candidates: Dict[str, List[MappingCandidate]] = {
        field: [] for field in CANONICAL_FIELDS
    }

    for canonical_field in CANONICAL_FIELDS:
        for source_column in dataframe.columns:
            source_column_text = str(source_column)

            score, reason = _candidate_score(
                source_column_text,
                canonical_field,
            )

            if score > 0:
                candidates[canonical_field].append(
                    MappingCandidate(
                        canonical_field=canonical_field,
                        source_column=source_column_text,
                        score=score,
                        reason=reason,
                    )
                )

        candidates[canonical_field].sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.source_column,
            )
        )

    return candidates


def map_columns(dataframe: pd.DataFrame) -> MappingResult:
    """
    Dynamically map source columns to the canonical schema.

    The mapper never fabricates missing information and never silently
    selects between equally strong candidates.
    """
    result = MappingResult()

    if dataframe is None or not isinstance(dataframe, pd.DataFrame):
        result.errors.append("Input must be a pandas DataFrame.")
        return result

    if dataframe.empty:
        result.errors.append("Input dataframe is empty.")
        return result

    candidates = _build_candidates(dataframe)

    # Detect ambiguity BEFORE assigning source columns.
    for canonical_field, field_candidates in candidates.items():
        if not field_candidates:
            continue

        best_score = field_candidates[0].score

        best_candidates = [
            candidate
            for candidate in field_candidates
            if candidate.score == best_score
        ]

        if len(best_candidates) > 1:
            result.ambiguous[canonical_field] = [
                candidate.source_column
                for candidate in best_candidates
            ]

    # Assign only fields that are not ambiguous.
    used_source_columns: set[str] = set()

    for canonical_field in CANONICAL_FIELDS:
        field_candidates = candidates[canonical_field]

        if not field_candidates:
            continue

        if canonical_field in result.ambiguous:
            continue

        selected: Optional[MappingCandidate] = None

        for candidate in field_candidates:
            if candidate.source_column not in used_source_columns:
                selected = candidate
                break

        if selected is None:
            result.warnings.append(
                f"No unused source column remained for "
                f"canonical field '{canonical_field}'."
            )
            continue

        result.mappings[canonical_field] = selected.source_column
        used_source_columns.add(selected.source_column)

    mapped_source_columns = set(result.mappings.values())

    result.unmapped_source_columns = [
        str(column)
        for column in dataframe.columns
        if str(column) not in mapped_source_columns
    ]

    result.missing_fields = [
        field
        for field in REQUIRED_CANONICAL_FIELDS
        if field not in result.mappings
    ]

    if result.missing_fields:
        result.warnings.append(
            "Missing required canonical fields: "
            + ", ".join(result.missing_fields)
        )

    return result


def normalize_canonical_dataframe(
    dataframe: pd.DataFrame,
    mappings: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalize a source dataframe into the canonical internal schema.

    Unmapped fields remain pd.NA. No missing business values are
    fabricated as zero.
    """
    canonical = pd.DataFrame(index=dataframe.index)
    warnings: List[str] = []

    for field in CANONICAL_FIELDS:
        source_column = mappings.get(field)

        if source_column is None:
            canonical[field] = pd.NA
            continue

        if source_column not in dataframe.columns:
            canonical[field] = pd.NA
            warnings.append(
                f"{field}: mapped source column '{source_column}' "
                f"was not found in the dataframe."
            )
            continue

        canonical[field] = dataframe[source_column]

    if "date" in canonical.columns:
        converted_date, date_warnings = safe_datetime(
            canonical["date"]
        )

        canonical["date"] = converted_date

        warnings.extend(
            [
                f"date: {message}"
                for message in _normalization_warnings(date_warnings)
            ]
        )

    for field in NUMERIC_CANONICAL_FIELDS:
        if field not in canonical.columns:
            continue

        converted_numeric, numeric_warnings = safe_numeric(
            canonical[field]
        )

        canonical[field] = converted_numeric

        warnings.extend(
            [
                f"{field}: {message}"
                for message in _normalization_warnings(numeric_warnings)
            ]
        )

    for field in STRING_CANONICAL_FIELDS:
        if field in canonical.columns:
            canonical[field] = clean_string_series(
                canonical[field]
            )

    return canonical, warnings


def create_mapping_result(
    dataframe: pd.DataFrame,
) -> MappingResult:
    """
    Run dynamic mapping and normalize the dataframe when mappings
    are sufficiently certain.
    """
    result = map_columns(dataframe)

    if result.errors:
        return result

    if result.ambiguous:
        result.warnings.append(
            "Ambiguous column mappings require user confirmation "
            "before normalization."
        )
        return result

    canonical_dataframe, conversion_warnings = (
        normalize_canonical_dataframe(
            dataframe,
            result.mappings,
        )
    )

    result.canonical_dataframe = canonical_dataframe

    result.conversion_warnings.extend(
        conversion_warnings
    )

    result.warnings.extend(
        conversion_warnings
    )

    return result


def describe_mapping(
    result: MappingResult,
) -> str:
    """
    Produce a human-readable mapping summary for the Streamlit UI.
    """
    lines: List[str] = []

    if result.mappings:
        lines.append("Confirmed mappings:")

        for canonical_field, source_column in result.mappings.items():
            lines.append(
                f"- {canonical_field} <- {source_column}"
            )

    if result.ambiguous:
        lines.append("")
        lines.append("Ambiguous mappings:")

        for canonical_field, source_columns in result.ambiguous.items():
            lines.append(
                f"- {canonical_field}: "
                + ", ".join(source_columns)
            )

    if result.missing_fields:
        lines.append("")
        lines.append("Missing required fields:")

        for field in result.missing_fields:
            lines.append(f"- {field}")

    if result.unmapped_source_columns:
        lines.append("")
        lines.append("Unmapped source columns:")

        for source_column in result.unmapped_source_columns:
            lines.append(f"- {source_column}")

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")

        for warning in result.warnings:
            lines.append(f"- {warning}")

    if result.conversion_warnings:
        lines.append("")
        lines.append("Conversion warnings:")

        for warning in result.conversion_warnings:
            lines.append(f"- {warning}")

    if result.errors:
        lines.append("")
        lines.append("Errors:")

        for error in result.errors:
            lines.append(f"- {error}")

    return "\n".join(lines)
