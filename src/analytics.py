from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


CANONICAL_ANALYTICS_FIELDS = [
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


def _numeric_series(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    """Return a numeric series if the column exists and contains usable values."""
    if column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return None

    return values


def _sum_column(df: pd.DataFrame, column: str) -> Optional[float]:
    """Return a deterministic sum, or None when the field is unavailable."""
    values = _numeric_series(df, column)
    if values is None:
        return None

    return float(values.sum())


def _weighted_or_row_cost(
    df: pd.DataFrame,
    quantity_column: str,
    cost_column: str,
) -> Optional[float]:
    """
    Calculate quantity × unit cost where both fields are available.

    Existing explicit cost columns are preferred by callers because they may
    already represent the actual recorded financial loss/rework cost.
    """
    quantity = _numeric_series(df, quantity_column)
    unit_cost = _numeric_series(df, "unit_cost")

    if quantity is None or unit_cost is None:
        return None

    valid = quantity.notna() & unit_cost.notna()
    if not valid.any():
        return None

    return float((quantity[valid] * unit_cost[valid]).sum())


def _explicit_or_derived_cost(
    df: pd.DataFrame,
    explicit_column: str,
    quantity_column: str,
) -> Optional[float]:
    """
    Use an explicit recorded cost when available.

    Otherwise derive the cost from quantity × unit cost. This is an
    illustrative deterministic calculation, not a verified accounting value.
    """
    explicit = _sum_column(df, explicit_column)
    if explicit is not None:
        return explicit

    return _weighted_or_row_cost(df, quantity_column, "unit_cost")


def _rate(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Calculate percentage safely, returning None when unavailable."""
    if numerator is None or denominator is None:
        return None

    if denominator == 0:
        return None

    return float((numerator / denominator) * 100.0)


def _group_sum(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Group a numeric metric by a dimension.

    Rows with unavailable grouping values are excluded rather than labeled
    with fabricated categories.
    """
    if group_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame(columns=[group_column, value_column])

    values = pd.to_numeric(df[value_column], errors="coerce")
    working = df[[group_column]].copy()
    working[value_column] = values

    working = working.dropna(subset=[group_column, value_column])

    if working.empty:
        return pd.DataFrame(columns=[group_column, value_column])

    result = (
        working.groupby(group_column, dropna=False)[value_column]
        .sum()
        .reset_index()
        .sort_values(value_column, ascending=False)
        .reset_index(drop=True)
    )

    return result


def _group_cost(
    df: pd.DataFrame,
    group_column: str,
    quantity_column: str,
) -> pd.DataFrame:
    """
    Calculate quantity × unit cost by dimension.

    Only rows with both a valid group and valid quantity/unit cost are used.
    """
    if (
        group_column not in df.columns
        or quantity_column not in df.columns
        or "unit_cost" not in df.columns
    ):
        return pd.DataFrame(columns=[group_column, "estimated_cost"])

    working = df[[group_column, quantity_column, "unit_cost"]].copy()

    working[quantity_column] = pd.to_numeric(
        working[quantity_column],
        errors="coerce",
    )
    working["unit_cost"] = pd.to_numeric(
        working["unit_cost"],
        errors="coerce",
    )

    working = working.dropna(
        subset=[group_column, quantity_column, "unit_cost"]
    )

    if working.empty:
        return pd.DataFrame(columns=[group_column, "estimated_cost"])

    working["estimated_cost"] = (
        working[quantity_column] * working["unit_cost"]
    )

    result = (
        working.groupby(group_column, dropna=False)["estimated_cost"]
        .sum()
        .reset_index()
        .sort_values("estimated_cost", ascending=False)
        .reset_index(drop=True)
    )

    return result


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate top-level deterministic WasteWise KPIs.

    Missing metrics are represented as None rather than zero.
    """
    production = _sum_column(df, "production_quantity")

    final_loss = _sum_column(df, "final_loss_quantity")

    # If final_loss_quantity is unavailable, loss_quantity is a valid fallback
    # for a dataset that records loss but not the final material-flow stage.
    if final_loss is None:
        final_loss = _sum_column(df, "loss_quantity")

    rework = _sum_column(df, "rework_quantity")
    recovered = _sum_column(df, "recovered_quantity")

    loss_cost = _explicit_or_derived_cost(
        df,
        explicit_column="loss_cost",
        quantity_column="final_loss_quantity",
    )

    if loss_cost is None:
        loss_cost = _explicit_or_derived_cost(
            df,
            explicit_column="loss_cost",
            quantity_column="loss_quantity",
        )

    rework_cost = _explicit_or_derived_cost(
        df,
        explicit_column="rework_cost",
        quantity_column="rework_quantity",
    )

    return {
        "total_production": production,
        "total_final_loss": final_loss,
        "loss_rate": _rate(final_loss, production),
        "loss_cost": loss_cost,
        "total_rework": rework,
        "total_recovered": recovered,
        "recovery_rate": _rate(recovered, rework),
        "rework_cost": rework_cost,
    }


def calculate_loss_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return deterministic loss analysis across the required dimensions."""
    return {
        "by_product": _group_sum(
            df,
            "product",
            "final_loss_quantity",
        )
        if "final_loss_quantity" in df.columns
        else _group_sum(df, "product", "loss_quantity"),
        "by_line": _group_sum(
            df,
            "production_line",
            "final_loss_quantity",
        )
        if "final_loss_quantity" in df.columns
        else _group_sum(df, "production_line", "loss_quantity"),
        "by_shift": _group_sum(
            df,
            "shift",
            "final_loss_quantity",
        )
        if "final_loss_quantity" in df.columns
        else _group_sum(df, "shift", "loss_quantity"),
        "by_event_type": _group_sum(
            df,
            "event_type",
            "final_loss_quantity",
        )
        if "final_loss_quantity" in df.columns
        else _group_sum(df, "event_type", "loss_quantity"),
        "by_reason": _group_sum(
            df,
            "event_reason",
            "final_loss_quantity",
        )
        if "final_loss_quantity" in df.columns
        else _group_sum(df, "event_reason", "loss_quantity"),
    }


def calculate_rework_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return deterministic rework analysis by product and production line."""
    return {
        "by_product": _group_sum(
            df,
            "product",
            "rework_quantity",
        ),
        "by_line": _group_sum(
            df,
            "production_line",
            "rework_quantity",
        ),
    }


def calculate_final_loss_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return final-loss analysis by product, line, and shift."""
    return {
        "by_product": _group_sum(
            df,
            "product",
            "final_loss_quantity",
        ),
        "by_line": _group_sum(
            df,
            "production_line",
            "final_loss_quantity",
        ),
        "by_shift": _group_sum(
            df,
            "shift",
            "final_loss_quantity",
        ),
    }


def calculate_loss_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate loss and production totals over time.

    Dates that cannot be parsed are excluded. No dates are fabricated.
    """
    if "date" not in df.columns:
        return pd.DataFrame(
            columns=[
                "date",
                "production_quantity",
                "final_loss_quantity",
                "loss_rate",
            ]
        )

    loss_column = (
        "final_loss_quantity"
        if "final_loss_quantity" in df.columns
        else "loss_quantity"
    )

    if loss_column not in df.columns:
        return pd.DataFrame(
            columns=[
                "date",
                "production_quantity",
                "final_loss_quantity",
                "loss_rate",
            ]
        )

    working = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "production_quantity": pd.to_numeric(
                df.get("production_quantity"),
                errors="coerce",
            ),
            "final_loss_quantity": pd.to_numeric(
                df[loss_column],
                errors="coerce",
            ),
        }
    )

    working = working.dropna(subset=["date"])

    if working.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "production_quantity",
                "final_loss_quantity",
                "loss_rate",
            ]
        )

    result = (
        working.groupby("date", as_index=False)[
            ["production_quantity", "final_loss_quantity"]
        ]
        .sum(min_count=1)
        .sort_values("date")
        .reset_index(drop=True)
    )

    result["loss_rate"] = np.where(
        result["production_quantity"].notna()
        & (result["production_quantity"] != 0),
        (
            result["final_loss_quantity"]
            / result["production_quantity"]
        )
        * 100.0,
        np.nan,
    )

    return result


def detect_anomalies(
    df: pd.DataFrame,
    group_column: str,
    value_column: str = "final_loss_quantity",
    z_threshold: float = 2.0,
    min_history: int = 3,
) -> pd.DataFrame:
    """
    Flag unusually high values using historical mean + standard deviation.

    The comparison is performed independently within each group.

    The current observation is compared against the group's historical
    observations, excluding the current row itself. This prevents the current
    event from changing its own baseline.

    Returns:
        A row-level dataframe containing:
        - original row index
        - group
        - observed value
        - historical mean
        - historical standard deviation
        - z_score
        - is_anomaly
        - explanation
    """
    required = {group_column, value_column}
    if not required.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "row_index",
                group_column,
                value_column,
                "historical_mean",
                "historical_std",
                "z_score",
                "is_anomaly",
                "explanation",
            ]
        )

    working = df[[group_column, value_column]].copy()
    working["_row_index"] = df.index
    working[value_column] = pd.to_numeric(
        working[value_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[group_column, value_column]
    )

    if working.empty:
        return pd.DataFrame(
            columns=[
                "row_index",
                group_column,
                value_column,
                "historical_mean",
                "historical_std",
                "z_score",
                "is_anomaly",
                "explanation",
            ]
        )

    rows: List[Dict[str, Any]] = []

    for _, row in working.iterrows():
        group_value = row[group_column]
        observed = float(row[value_column])

        history = working[
            (working[group_column] == group_value)
            & (working["_row_index"] != row["_row_index"])
        ][value_column]

        history = pd.to_numeric(history, errors="coerce").dropna()

        if len(history) < min_history:
            rows.append(
                {
                    "row_index": row["_row_index"],
                    group_column: group_value,
                    value_column: observed,
                    "historical_mean": None,
                    "historical_std": None,
                    "z_score": None,
                    "is_anomaly": False,
                    "explanation": (
                        f"Insufficient historical observations for "
                        f"{group_value!r}; anomaly baseline unavailable."
                    ),
                }
            )
            continue

        historical_mean = float(history.mean())
        historical_std = float(history.std(ddof=0))

        if historical_std == 0:
            is_anomaly = (
                observed > historical_mean
                and observed != historical_mean
            )
            z_score = None

            if is_anomaly:
                explanation = (
                    f"{group_value!r} is above a constant historical "
                    f"baseline of {historical_mean:.3f}."
                )
            else:
                explanation = (
                    f"{group_value!r} is not above its historical baseline."
                )
        else:
            z_score = float(
                (observed - historical_mean) / historical_std
            )
            is_anomaly = bool(z_score >= z_threshold)

            explanation = (
                f"Observed {observed:.3f}; historical mean "
                f"{historical_mean:.3f}; {z_score:.2f} standard deviations "
                f"from the historical mean."
            )

        rows.append(
            {
                "row_index": row["_row_index"],
                group_column: group_value,
                value_column: observed,
                "historical_mean": historical_mean,
                "historical_std": historical_std,
                "z_score": z_score,
                "is_anomaly": bool(is_anomaly),
                "explanation": explanation,
            }
        )

    return pd.DataFrame(rows)


def calculate_financial_impact(
    df: pd.DataFrame,
    illustrative_reduction_rate: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Calculate deterministic estimated loss cost and optional illustrative
    savings scenarios.

    `illustrative_reduction_rate` must be supplied as a percentage, e.g. 10
    for a 10% scenario.

    These values are estimates unless verified against actual factory costs.
    """
    kpis = calculate_kpis(df)

    estimated_loss_cost = kpis["loss_cost"]

    result: Dict[str, Optional[float]] = {
        "estimated_waste_cost": estimated_loss_cost,
        "illustrative_monthly_saving": None,
        "illustrative_annualized_value": None,
    }

    if (
        illustrative_reduction_rate is not None
        and estimated_loss_cost is not None
    ):
        reduction_fraction = float(illustrative_reduction_rate) / 100.0

        monthly_saving = estimated_loss_cost * reduction_fraction

        result["illustrative_monthly_saving"] = float(monthly_saving)
        result["illustrative_annualized_value"] = float(
            monthly_saving * 12.0
        )

    return result


def rank_investigation_priorities(
    df: pd.DataFrame,
    group_column: str = "production_line",
) -> pd.DataFrame:
    """
    Rank groups using deterministic loss quantity and loss rate.

    Priority is intentionally an investigation indicator, not a root-cause
    conclusion.
    """
    if group_column not in df.columns:
        return pd.DataFrame(
            columns=[
                group_column,
                "production_quantity",
                "final_loss_quantity",
                "loss_rate",
                "estimated_loss_cost",
            ]
        )

    loss_column = (
        "final_loss_quantity"
        if "final_loss_quantity" in df.columns
        else "loss_quantity"
    )

    required = {
        group_column,
        "production_quantity",
        loss_column,
    }

    if not required.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                group_column,
                "production_quantity",
                "final_loss_quantity",
                "loss_rate",
                "estimated_loss_cost",
            ]
        )

    production = _group_sum(
        df,
        group_column,
        "production_quantity",
    ).rename(
        columns={"production_quantity": "production_quantity"}
    )

    loss = _group_sum(
        df,
        group_column,
        loss_column,
    ).rename(
        columns={loss_column: "final_loss_quantity"}
    )

    result = production.merge(
        loss,
        on=group_column,
        how="outer",
    )

    if result.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "production_quantity",
                "final_loss_quantity",
                "loss_rate",
                "estimated_loss_cost",
            ]
        )

    result["loss_rate"] = np.where(
        result["production_quantity"].notna()
        & (result["production_quantity"] != 0),
        (
            result["final_loss_quantity"]
            / result["production_quantity"]
        )
        * 100.0,
        np.nan,
    )

    cost_df = _group_cost(
        df,
        group_column,
        loss_column,
    ).rename(
        columns={"estimated_cost": "estimated_loss_cost"}
    )

    result = result.merge(
        cost_df,
        on=group_column,
        how="left",
    )

    result = result.sort_values(
        by=["estimated_loss_cost", "loss_rate", "final_loss_quantity"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return result


def build_analytics_context(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a compact deterministic analytics object for later Dashboard and
    AI Analyst stages.
    """
    return {
        "kpis": calculate_kpis(df),
        "loss_analysis": calculate_loss_analysis(df),
        "rework_analysis": calculate_rework_analysis(df),
        "final_loss_analysis": calculate_final_loss_analysis(df),
        "loss_trend": calculate_loss_trend(df),
        "financial_impact": calculate_financial_impact(df),
        "line_priorities": rank_investigation_priorities(
            df,
            group_column="production_line",
        ),
    }
