import numpy as np
import pandas as pd

from src.analytics import (
    build_analytics_context,
    calculate_financial_impact,
    calculate_kpis,
    calculate_loss_analysis,
    calculate_loss_trend,
    calculate_rework_analysis,
    detect_anomalies,
    rank_investigation_priorities,
)


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
                "2026-01-03",
            ],
            "product": [
                "Product A",
                "Product B",
                "Product A",
                "Product B",
                "Product A",
            ],
            "batch": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
            ],
            "production_line": [
                "Line 1",
                "Line 2",
                "Line 1",
                "Line 2",
                "Line 1",
            ],
            "shift": [
                "Day",
                "Night",
                "Day",
                "Night",
                "Day",
            ],
            "production_quantity": [
                1000,
                1000,
                1200,
                1000,
                800,
            ],
            "event_type": [
                "Packaging Loss",
                "Process Loss",
                "Quality Deviation",
                "Packaging Loss",
                "Process Loss",
            ],
            "event_reason": [
                "Seal",
                "Trim",
                "Quality Check",
                "Seal",
                "Overfill",
            ],
            "affected_quantity": [
                30,
                20,
                25,
                50,
                40,
            ],
            "loss_quantity": [
                20,
                15,
                10,
                40,
                30,
            ],
            "rework_quantity": [
                5,
                10,
                15,
                5,
                10,
            ],
            "recovered_quantity": [
                2,
                8,
                10,
                2,
                7,
            ],
            "final_loss_quantity": [
                18,
                7,
                0,
                38,
                23,
            ],
            "disposition": [
                "Disposed",
                "Reworked",
                "Recovered",
                "Disposed",
                "Disposed",
            ],
            "unit_cost": [
                100,
                100,
                100,
                100,
                100,
            ],
            "rework_cost": [
                50,
                100,
                150,
                50,
                100,
            ],
            "loss_cost": [
                1800,
                700,
                0,
                3800,
                2300,
            ],
        }
    )


def test_kpis_are_deterministic():
    df = sample_dataframe()

    kpis = calculate_kpis(df)

    assert kpis["total_production"] == 5000
    assert kpis["total_final_loss"] == 86

    expected_loss_rate = (86 / 5000) * 100
    assert np.isclose(
        kpis["loss_rate"],
        expected_loss_rate,
    )

    assert kpis["loss_cost"] == 8600
    assert kpis["total_rework"] == 45
    assert kpis["total_recovered"] == 29

    expected_recovery_rate = (29 / 45) * 100
    assert np.isclose(
        kpis["recovery_rate"],
        expected_recovery_rate,
    )


def test_loss_analysis_by_product():
    df = sample_dataframe()

    analysis = calculate_loss_analysis(df)
    result = analysis["by_product"]

    assert list(result.columns) == [
        "product",
        "final_loss_quantity",
    ]

    product_a = result[
        result["product"] == "Product A"
    ]["final_loss_quantity"].iloc[0]

    product_b = result[
        result["product"] == "Product B"
    ]["final_loss_quantity"].iloc[0]

    assert product_a == 41
    assert product_b == 45


def test_loss_analysis_by_line():
    df = sample_dataframe()

    analysis = calculate_loss_analysis(df)
    result = analysis["by_line"]

    line_1 = result[
        result["production_line"] == "Line 1"
    ]["final_loss_quantity"].iloc[0]

    line_2 = result[
        result["production_line"] == "Line 2"
    ]["final_loss_quantity"].iloc[0]

    assert line_1 == 41
    assert line_2 == 45


def test_rework_analysis():
    df = sample_dataframe()

    analysis = calculate_rework_analysis(df)

    by_product = analysis["by_product"]

    product_a = by_product[
        by_product["product"] == "Product A"
    ]["rework_quantity"].iloc[0]

    product_b = by_product[
        by_product["product"] == "Product B"
    ]["rework_quantity"].iloc[0]

    assert product_a == 30
    assert product_b == 15


def test_loss_trend():
    df = sample_dataframe()

    trend = calculate_loss_trend(df)

    assert len(trend) == 3
    assert list(trend["date"].dt.strftime("%Y-%m-%d")) == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]

    assert trend.loc[0, "production_quantity"] == 2000
    assert trend.loc[0, "final_loss_quantity"] == 25

    assert np.isclose(
        trend.loc[0, "loss_rate"],
        1.25,
    )


def test_financial_impact_with_illustrative_scenario():
    df = sample_dataframe()

    impact = calculate_financial_impact(
        df,
        illustrative_reduction_rate=10,
    )

    assert impact["estimated_waste_cost"] == 8600
    assert impact["illustrative_monthly_saving"] == 860
    assert impact["illustrative_annualized_value"] == 10320


def test_financial_impact_without_scenario_does_not_invent_savings():
    df = sample_dataframe()

    impact = calculate_financial_impact(df)

    assert impact["estimated_waste_cost"] == 8600
    assert impact["illustrative_monthly_saving"] is None
    assert impact["illustrative_annualized_value"] is None


def test_missing_metrics_remain_unavailable():
    df = pd.DataFrame(
        {
            "production_quantity": [100, 200],
        }
    )

    kpis = calculate_kpis(df)

    assert kpis["total_production"] == 300
    assert kpis["total_final_loss"] is None
    assert kpis["loss_rate"] is None
    assert kpis["loss_cost"] is None
    assert kpis["total_rework"] is None
    assert kpis["total_recovered"] is None
    assert kpis["recovery_rate"] is None


def test_missing_rework_does_not_become_zero():
    df = pd.DataFrame(
        {
            "production_quantity": [100, 100],
            "final_loss_quantity": [10, 20],
        }
    )

    kpis = calculate_kpis(df)

    assert kpis["total_rework"] is None
    assert kpis["total_recovered"] is None
    assert kpis["recovery_rate"] is None


def test_explicit_loss_cost_is_preferred():
    df = pd.DataFrame(
        {
            "final_loss_quantity": [10, 20],
            "unit_cost": [100, 100],
            "loss_cost": [1500, 2500],
        }
    )

    kpis = calculate_kpis(df)

    assert kpis["loss_cost"] == 4000


def test_loss_cost_can_be_derived_from_quantity_and_unit_cost():
    df = pd.DataFrame(
        {
            "final_loss_quantity": [10, 20],
            "unit_cost": [100, 150],
        }
    )

    kpis = calculate_kpis(df)

    assert kpis["loss_cost"] == 4000


def test_anomaly_detection_flags_high_observation():
    df = pd.DataFrame(
        {
            "production_line": [
                "Line 1",
                "Line 1",
                "Line 1",
                "Line 1",
            ],
            "final_loss_quantity": [
                10,
                11,
                9,
                30,
            ],
        }
    )

    anomalies = detect_anomalies(
        df,
        group_column="production_line",
        value_column="final_loss_quantity",
        z_threshold=2.0,
        min_history=3,
    )

    high_row = anomalies[
        anomalies["row_index"] == 3
    ].iloc[0]

    assert high_row["is_anomaly"] is True
    assert high_row["z_score"] is not None
    assert high_row["z_score"] > 2.0


def test_anomaly_detection_requires_history():
    df = pd.DataFrame(
        {
            "production_line": [
                "Line 1",
                "Line 1",
            ],
            "final_loss_quantity": [
                10,
                30,
            ],
        }
    )

    anomalies = detect_anomalies(
        df,
        group_column="production_line",
        value_column="final_loss_quantity",
        min_history=3,
    )

    assert len(anomalies) == 2
    assert anomalies["is_anomaly"].sum() == 0
    assert anomalies["historical_mean"].isna().all()


def test_anomaly_detection_does_not_claim_root_cause():
    df = pd.DataFrame(
        {
            "production_line": [
                "Line 1",
                "Line 1",
                "Line 1",
                "Line 1",
            ],
            "final_loss_quantity": [
                10,
                11,
                9,
                30,
            ],
        }
    )

    anomalies = detect_anomalies(
        df,
        group_column="production_line",
        value_column="final_loss_quantity",
        min_history=3,
    )

    explanation = anomalies.loc[
        anomalies["row_index"] == 3,
        "explanation",
    ].iloc[0]

    assert "standard deviations" in explanation
    assert "root cause" not in explanation.lower()


def test_priority_ranking_is_deterministic():
    df = sample_dataframe()

    priorities = rank_investigation_priorities(
        df,
        group_column="production_line",
    )

    assert list(priorities.columns) == [
        "production_line",
        "production_quantity",
        "final_loss_quantity",
        "loss_rate",
        "estimated_loss_cost",
    ]

    assert priorities.iloc[0]["production_line"] == "Line 2"
    assert priorities.iloc[0]["final_loss_quantity"] == 45
    assert priorities.iloc[0]["estimated_loss_cost"] == 4500


def test_priority_ranking_can_use_loss_quantity_fallback():
    df = pd.DataFrame(
        {
            "production_line": ["Line A", "Line A", "Line B"],
            "production_quantity": [100, 100, 100],
            "loss_quantity": [10, 20, 5],
            "unit_cost": [10, 10, 10],
        }
    )

    priorities = rank_investigation_priorities(
        df,
        group_column="production_line",
    )

    assert priorities.iloc[0]["production_line"] == "Line A"
    assert priorities.iloc[0]["final_loss_quantity"] == 30
    assert priorities.iloc[0]["estimated_loss_cost"] == 300


def test_build_analytics_context_contains_required_sections():
    df = sample_dataframe()

    context = build_analytics_context(df)

    assert "kpis" in context
    assert "loss_analysis" in context
    assert "rework_analysis" in context
    assert "final_loss_analysis" in context
    assert "loss_trend" in context
    assert "financial_impact" in context
    assert "line_priorities" in context

    assert context["kpis"]["total_production"] == 5000
