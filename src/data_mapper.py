"""
WasteWise AI dynamic data mapping and normalization.

The mapper converts factory-specific source column names into the
WasteWise canonical schema.

Important:
    - It does not assume a specific factory.
    - It does not silently fabricate missing fields.
    - Ambiguous mappings are explicitly reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .utils import (
    clean_string_series,
    normalize_column_name,
    normalize_text,
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


REQUIRED_FIELDS = [
    "date",
    "product",
    "production_quantity",
]


FIELD_ALIASES = {
    "date": [
        "date",
        "production date",
        "prod date",
        "event date",
        "record date",
        "transaction date",
    ],
    "product": [
        "product",
        "product name",
        "product code",
        "sku",
        "item",
        "item name",
    ],
    "batch": [
        "batch",
        "batch id",
        "batch number",
        "batch no",
        "lot",
        "lot number",
    ],
    "production_line": [
        "production line",
        "production_line",
        "line",
        "line id",
        "line name",
        "prod line",
        "production unit",
    ],
    "shift": [
        "shift",
        "shift name",
        "work shift",
        "production shift",
    ],
    "production_quantity": [
        "production quantity",
        "production qty",
        "prod quantity",
        "prod qty",
        "production volume",
        "production output",
        "output quantity",
        "output qty",
        "produced quantity",
        "produced qty",
    ],
    "event_type": [
        "event type",
        "event_type",
        "loss type",
        "waste type",
        "incident type",
        "record type",
    ],
    "event_reason": [
        "event reason",
        "reason",
        "waste reason",
        "loss reason",
        "cause",
        "loss cause",
        "waste cause",
        "reason description",
    ],
    "affected_quantity": [
        "affected quantity",
        "affected qty",
        "affected volume",
        "quantity affected",
    ],
    "loss_quantity": [
        "loss quantity",
        "loss qty",
        "waste quantity",
        "waste qty",
        "waste",
        "scrap quantity",
        "scrap qty",
        "material loss",
        "loss volume",
    ],
    "rework_quantity": [
        "rework quantity",
        "rework qty",
        "rework volume",
        "quantity reworked",
    ],
    "recovered_quantity": [
        "recovered quantity",
        "recovered qty",
        "recovery quantity",
        "recovery qty",
        "recovered volume",
    ],
    "final_loss_quantity": [
        "final loss quantity",
        "final loss qty",
        "final waste quantity",
        "final waste qty",
        "net loss quantity",
        "disposal quantity",
        "disposed quantity",
    ],
    "disposition": [
        "disposition",
        "disposition status",
        "material disposition",
        "outcome",
        "status",
    ],
    "unit_cost": [
        "unit cost",
        "cost per unit",
        "unit price",
        "material unit cost",
        "cost unit",
    ],
    "rework_cost": [
        "rework cost",
        "cost of rework",
        "rework expense",
    ],
    "loss_cost": [
        "loss cost",
        "waste cost",
        "scrap cost",
        "cost of loss",
        "disposal cost",
    ],
    "quality_parameter": [
        "quality parameter",
        "quality metric",
        "test parameter",
        "parameter",
        "qc parameter",
        "quality test",
    ],
    "quality_value": [
        "quality value",
        "test value",
        "measurement",
        "measured value",
        "qc value",
        "result value",
    ],
    "quality_status": [
        "quality status",
        "qc status",
        "quality result",
        "qc result",
        "conformance",
        "conformance status",
        "spec status",
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
class MappingCandidate:
    """A possible source-to-canonical mapping."""

    source_column: str
    canonical_field: str
    match_type: str
    score: float


@dataclass
class MappingResult:
    """Structured result of the dynamic mapping process."""

    canonical_dataframe: pd.DataFrame | None
    mappings: dict[str, str] = field(default_factory=dict)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    unmapped_source_columns: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    conversion_warnings: list[str] = field(default_factory=list)


def _normalized_aliases(canonical_field: str) -> set[str]:
    """Return normalized aliases for a canonical field."""
    aliases = FIELD_ALIASES.get(
        canonical_field,
        [],
    )

    return {
        normalize_column_name(alias)
        for alias in aliases
    }


def _candidate_score(
    source_column: str,
    canonical_field: str,
) -> tuple[str, float]:
    """
    Determine how strongly a source column matches a canonical field.

    Matching order:
        exact canonical name
        exact alias
        token/substring relationship
        no match
    """
    normalized_source = normalize_column_name(
        source_column
    )

    normalized_canonical = normalize_column_name(
        canonical_field
    )

    aliases = _normalized_aliases(
        canonical_field
    )

    if normalized_source == normalized_canonical:
        return "EXACT", 1.00

    if normalized_source in aliases:
        return "ALIAS", 0.95

    source_tokens = set(
        normalized_source.split("_")
    )

    for alias in aliases:
        alias_tokens = set(
            alias.split("_")
        )

        if not alias_tokens:
            continue

        overlap = len(
            source_tokens & alias_tokens
        ) / max(
            len(alias_tokens),
            1,
        )

        if overlap >= 0.75:
            return "TOKEN", 0.80

    return "NONE", 0.0


def generate_candidates(
    columns: list[str],
) -> list[MappingCandidate]:
    """Generate all meaningful source-column mapping candidates."""
    candidates: list[MappingCandidate] = []

    for source_column in columns:
        for canonical_field in CANONICAL_FIELDS:
            match_type, score = _candidate_score(
                source_column,
                canonical_field,
            )

            if score > 0:
                candidates.append(
                    MappingCandidate(
                        source_column=source_column,
                        canonical_field=canonical_field,
                        match_type=match_type,
                        score=score,
                    )
                )

    return candidates


def map_columns(
    dataframe: pd.DataFrame,
) -> MappingResult:
    """
    Dynamically map source dataframe columns to canonical fields.

    Ambiguous fields are not automatically assigned.
    """
    columns = [
        str(column)
        for column in dataframe.columns
    ]

    candidates = generate_candidates(
        columns
    )

    grouped: dict[str, list[MappingCandidate]] = {
        field: []
        for field in CANONICAL_FIELDS
    }

    for candidate in candidates:
        grouped[
            candidate.canonical_field
        ].append(candidate)

    mappings: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    used_source_columns: set[str] = set()

    for canonical_field in CANONICAL_FIELDS:
        field_candidates = sorted(
            grouped[canonical_field],
            key=lambda item: (
                item.score,
                item.match_type == "EXACT",
            ),
            reverse=True,
        )

        if not field_candidates:
            continue

        best_score = field_candidates[0].score

        best = [
            candidate
            for candidate in field_candidates
            if candidate.score == best_score
        ]

        if len(best) == 1:
            source = best[0].source_column

            if source not in used_source_columns:
                mappings[canonical_field] = source
                used_source_columns.add(source)
            else:
                ambiguous[canonical_field] = [
                    candidate.source_column
                    for candidate in best
                ]

        else:
            # Do not silently guess between equally strong candidates.
            ambiguous[canonical_field] = [
                candidate.source_column
                for candidate in best
            ]

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in mappings
    ]

    mapped_sources = set(
        mappings.values()
    )

    unmapped_source_columns = [
        column
        for column in columns
        if column not in mapped_sources
    ]

    warnings: list[str] = []

    for field, candidates_for_field in ambiguous.items():
        warnings.append(
            f"Ambiguous mapping for '{field}': "
            + ", ".join(candidates_for_field)
            + ". User confirmation is required."
        )

    for field in missing_fields:
        warnings.append(
            f"Required field '{field}' could not be mapped. "
            "WasteWise AI will not fabricate this field."
        )

    return MappingResult(
        canonical_dataframe=None,
        mappings=mappings,
        ambiguous=ambiguous,
        unmapped_source_columns=unmapped_source_columns,
        missing_fields=missing_fields,
        warnings=warnings,
    )


def normalize_canonical_dataframe(
    dataframe: pd.DataFrame,
    mappings: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create a canonical dataframe using the confirmed mappings.

    Missing canonical fields are left as pd.NA.
    They are never replaced with zero.
    """
    warnings: list[str] = []

    canonical = pd.DataFrame(
        index=dataframe.index
    )

    for field in CANONICAL_FIELDS:
        source_column = mappings.get(field)

        if source_column is None:
            canonical[field] = pd.NA
            continue

        if source_column not in dataframe.columns:
            canonical[field] = pd.NA
            warnings.append(
                f"Source column '{source_column}' "
                f"for '{field}' was not found."
            )
            continue

        canonical[field] = dataframe[
            source_column
        ]

    if "date" in mappings:
        converted, invalid_count = safe_datetime(
            canonical["date"]
        )
        canonical["date"] = converted

        if invalid_count:
            warnings.append(
                f"{invalid_count} value(s) in 'date' "
                "could not be converted to a valid date."
            )

    for field in NUMERIC_FIELDS:
        if field not in canonical.columns:
            continue

        converted, invalid_count = safe_numeric(
            canonical[field]
        )

        canonical[field] = converted

        if invalid_count:
            warnings.append(
                f"{invalid_count} value(s) in '{field}' "
                "could not be converted to numeric values."
            )

    for field in STRING_FIELDS:
        if field in canonical.columns:
            canonical[field] = clean_string_series(
                canonical[field]
            )

    return canonical, warnings


def create_mapping_result(
    dataframe: pd.DataFrame,
) -> MappingResult:
    """
    Run mapping and normalization in one operation.

    This is the main public entry point for the mapping layer.
    """
    mapping_result = map_columns(
        dataframe
    )

    # Do not normalize through unresolved ambiguous mappings.
    confirmed_mappings = dict(
        mapping_result.mappings
    )

    canonical, conversion_warnings = (
        normalize_canonical_dataframe(
            dataframe,
            confirmed_mappings,
        )
    )

    mapping_result.canonical_dataframe = canonical
    mapping_result.conversion_warnings = (
        conversion_warnings
    )
    mapping_result.warnings.extend(
        conversion_warnings
    )

    return mapping_result


def describe_mapping(
    result: MappingResult,
) -> dict[str, object]:
    """
    Convert MappingResult into a UI-friendly dictionary.
    """
    return {
        "mappings": result.mappings,
        "ambiguous": result.ambiguous,
        "missing_fields": result.missing_fields,
        "unmapped_source_columns": (
            result.unmapped_source_columns
        ),
        "warnings": result.warnings,
        "errors": result.errors,
    }
