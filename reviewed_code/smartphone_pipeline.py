"""
smartphone_pipeline.py
======================
ETL pipeline for smartphone sales data.

Stages: load → validate → clean → transform → compute KPIs → charts → save

Usage:
    python reviewed_code/smartphone_pipeline.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend, safe for scripts and CI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Paths are resolved relative to this file so the pipeline works on any machine.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "smartphone_sales.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHARTS_DIR    = PROJECT_ROOT / "reports" / "charts"


# Business validation boundaries.
# These live here so any change is immediately visible and doesn't require
# hunting through function bodies.

REQUIRED_COLUMNS: set[str] = {
    "order_id", "order_date", "customer_id", "country",
    "brand", "model", "price", "quantity", "discount",
    "payment_method", "rating", "returned",
}

MIN_VALID_PRICE: float = 1.0
MAX_VALID_PRICE: float = 9_999.0  # $99,999 outliers are excluded as data errors

MIN_VALID_QTY: int = 1
MAX_VALID_QTY: int = 100

MIN_RATING: float = 1.0
MAX_RATING: float = 5.0

MIN_DISCOUNT: float = 0.0
MAX_DISCOUNT: float = 0.999  # a 100 % discount is treated as invalid

# Country alias table: maps known dirty variants to canonical names.
# Covers casing differences, abbreviations, and typos seen in the source data.
COUNTRY_ALIASES: dict[str, str] = {
    "india":          "India",
    "inda":           "India",
    "usa":            "USA",
    "u.s.a":          "USA",
    "united states":  "USA",
    "uk":             "UK",
    "u.k.":           "UK",
    "united kingdom": "UK",
    "germany":        "Germany",
    "france":         "France",
    "japan":          "Japan",
    "australia":      "Australia",
    "canada":         "Canada",
    "brazil":         "Brazil",
    "singapore":      "Singapore",
}

# String values that mean "this order was returned".
TRUTHY_RETURN_VALUES: frozenset = frozenset({"yes", "true", "1"})

CHART_DPI: int = 150


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_data(path: pathlib.Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and validate that required columns are present.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame with all original columns preserved.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Run generate_dataset.py first to create the dataset."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path.name)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing_cols)}\n"
            f"Found: {sorted(df.columns)}"
        )

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_raw(df: pd.DataFrame) -> dict[str, int]:
    """Audit the raw DataFrame and log a data-quality summary.

    Does not modify the DataFrame. Intended to give the data team visibility
    into what the cleaning step will address before any rows are dropped.

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Dictionary mapping issue name to count of affected rows.
    """
    issues: dict[str, int] = {}

    issues["duplicate_rows"]     = int(df.duplicated().sum())
    issues["missing_order_date"] = int(df["order_date"].isna().sum())
    issues["missing_country"]    = int(df["country"].isna().sum())
    issues["missing_brand"]      = int(df["brand"].isna().sum())
    issues["missing_price"]      = int(df["price"].isna().sum())
    issues["missing_quantity"]   = int(df["quantity"].isna().sum())
    issues["missing_returned"]   = int(df["returned"].isna().sum())
    issues["missing_rating"]     = int(df["rating"].isna().sum())

    price_num  = pd.to_numeric(df["price"],    errors="coerce")
    qty_num    = pd.to_numeric(df["quantity"], errors="coerce")
    rating_num = pd.to_numeric(df["rating"],   errors="coerce")

    issues["invalid_price"]    = int((price_num <= 0).sum())
    issues["extreme_price"]    = int((price_num > MAX_VALID_PRICE).sum())
    issues["invalid_quantity"] = int(
        ((qty_num < MIN_VALID_QTY) | (qty_num > MAX_VALID_QTY)).sum()
    )
    issues["invalid_rating"] = int(
        ((rating_num < MIN_RATING) | (rating_num > MAX_RATING)).sum()
    )

    total = sum(issues.values())
    logger.info("Data quality audit — %d total issues found:", total)
    for name, count in issues.items():
        if count > 0:
            logger.warning("  %-25s : %d rows", name, count)

    return issues


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _normalise_returned(value) -> Optional[bool]:
    """Map a mixed-type 'returned' cell to True, False, or None.

    The source data stores this field as booleans, strings, and integers.
    None is returned for missing values — we don't assume "not returned"
    just because the field is absent.
    """
    if pd.isna(value):
        return None
    return str(value).strip().lower() in TRUTHY_RETURN_VALUES


def _normalise_country(raw: str) -> str:
    """Resolve a raw country string to its canonical form.

    Strips whitespace, lower-cases, then looks up COUNTRY_ALIASES.
    Falls back to title-casing the original value if no alias is found.
    """
    return COUNTRY_ALIASES.get(raw.strip().lower(), raw.strip().title())


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise the raw DataFrame.

    Steps applied in order:
    1.  Drop exact duplicate rows.
    2.  Drop rows missing fields required for any revenue or return analysis
        (order_id, price, quantity, returned). Optional fields like rating
        and payment_method are kept even when null.
    3.  Fill optional nulls with safe defaults.
    4.  Parse order_date; malformed values become NaT rather than crashing.
    5.  Normalise country names via alias table.
    6.  Normalise brand names to title case.
    7.  Map the returned column to bool / None.
    8.  Cast quantity to a nullable integer type.
    9.  Remove rows with out-of-range prices, quantities, discounts, ratings.

    Args:
        df: Raw DataFrame from load_data(). Not mutated.

    Returns:
        Cleaned DataFrame with reset index.
    """
    df = df.copy()
    original_len = len(df)

    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", original_len - len(df))

    # price, quantity, and returned are required for the core KPIs.
    # Dropping on these fields only preserves rows where other optional
    # columns (rating, payment_method) happen to be null.
    critical_cols = ["order_id", "price", "quantity", "returned"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    logger.info(
        "Dropped %d rows missing critical fields (%s)",
        before - len(df),
        critical_cols,
    )

    df["discount"]       = df["discount"].fillna(0.0)
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    # format="mixed" handles the multiple date formats present in the source
    # export without triggering a per-element fallback warning.
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")
    n_bad_dates = df["order_date"].isna().sum()
    if n_bad_dates > 0:
        logger.warning(
            "%d order_date values could not be parsed — set to NaT", n_bad_dates
        )

    df["country"]  = df["country"].fillna("Unknown").apply(_normalise_country)
    df["brand"]    = df["brand"].fillna("Unknown").str.strip().str.title()
    df["returned"] = df["returned"].map(_normalise_returned)

    # Int64 (nullable integer) preserves NaN rather than raising on mixed types.
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")

    before = len(df)
    df = df[df["price"].between(MIN_VALID_PRICE, MAX_VALID_PRICE)]
    df = df[df["quantity"].between(MIN_VALID_QTY, MAX_VALID_QTY)]
    df = df[df["discount"].between(MIN_DISCOUNT, MAX_DISCOUNT)]
    # Rating is optional — NaN rows are kept, only out-of-range values removed.
    rating_ok = df["rating"].isna() | df["rating"].between(MIN_RATING, MAX_RATING)
    df = df[rating_ok]
    logger.info(
        "Dropped %d rows with out-of-range prices, quantities, discounts, or ratings",
        before - len(df),
    )

    df = df.reset_index(drop=True)
    logger.info("Clean dataset: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue and date columns derived from the cleaned data.

    Columns added:
        discount_amount  dollar value of the discount per unit
        net_price        unit price after discount
        revenue          net_price × quantity (realised revenue per order)
        year             calendar year of the order
        month            calendar month (1–12)
        year_month       'YYYY-MM' string for time-series grouping

    Args:
        df: Cleaned DataFrame from clean_data(). Not mutated.

    Returns:
        DataFrame with derived columns appended.
    """
    df = df.copy()

    df["discount_amount"] = df["price"] * df["discount"]
    df["net_price"]       = df["price"] * (1.0 - df["discount"])
    df["revenue"]         = df["net_price"] * df["quantity"].astype(float)

    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    return df


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------

def calculate_return_rate(df: pd.DataFrame) -> float:
    """Return the fraction of orders that were returned.

    Rows where returned is None are excluded from both numerator and
    denominator. Using len(df) as the denominator would understate the rate
    when return status is unknown for some orders.

    Args:
        df: DataFrame with a boolean/None 'returned' column.

    Returns:
        Return rate in [0.0, 1.0]. Returns 0.0 if no valid data exists.
    """
    valid = df["returned"].dropna()
    if valid.empty:
        logger.warning("No valid 'returned' data — return rate set to 0.0")
        return 0.0
    return float(valid.sum() / len(valid))


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute top-level business KPIs from the enriched DataFrame.

    Args:
        df: Enriched DataFrame from add_derived_columns().

    Returns:
        Dictionary of KPI name → value.
    """
    kpis = {
        "total_orders":        len(df),
        "total_revenue":       round(float(df["revenue"].sum()), 2),
        "average_order_value": round(float(df["revenue"].mean()), 2),
        "return_rate_pct":     round(calculate_return_rate(df) * 100, 2),
        "average_rating":      round(float(df["rating"].mean()), 2),
        "unique_customers":    int(df["customer_id"].nunique()),
        "unique_brands":       int(df["brand"].nunique()),
        "unique_countries":    int(df["country"].nunique()),
    }
    logger.info("KPIs computed:")
    for k, v in kpis.items():
        logger.info("  %-28s : %s", k, v)
    return kpis


def revenue_by_brand(df: pd.DataFrame) -> pd.Series:
    """Return total revenue grouped by brand, sorted descending.

    Args:
        df: Enriched DataFrame with a 'revenue' column.

    Returns:
        Series indexed by brand name with total revenue values.
    """
    return (
        df.groupby("brand")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .rename("total_revenue")
    )


def revenue_by_country(df: pd.DataFrame) -> pd.Series:
    """Return total revenue grouped by country, sorted descending."""
    return (
        df.groupby("country")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .rename("total_revenue")
    )


def monthly_revenue(df: pd.DataFrame) -> pd.Series:
    """Return total revenue grouped by year-month, sorted chronologically.

    Rows with NaT order_date are excluded — they have no meaningful time period.

    Args:
        df: Enriched DataFrame with 'year_month' and 'revenue' columns.

    Returns:
        Series indexed by 'YYYY-MM' strings.
    """
    return (
        df.dropna(subset=["order_date"])
        .groupby("year_month")["revenue"]
        .sum()
        .sort_index()
        .rename("total_revenue")
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure to the charts directory and close it."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_DIR / filename
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", output_path.relative_to(PROJECT_ROOT))


def generate_charts(df: pd.DataFrame) -> None:
    """Generate and save four business charts to reports/charts/.

    Charts: revenue by brand (bar), monthly revenue trend (line),
    rating distribution (histogram), top-5 countries by revenue (pie).

    Args:
        df: Enriched DataFrame from add_derived_columns().
    """
    # Revenue by brand
    brand_rev = revenue_by_brand(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(brand_rev.index[::-1], brand_rev.values[::-1], color="#4C72B0")
    ax.set_xlabel("Total Revenue (USD)", fontsize=12)
    ax.set_title("Revenue by Brand", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="$%.0f", padding=4, fontsize=9)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    fig.tight_layout()
    _save_figure(fig, "brand_revenue.png")

    # Monthly revenue trend
    monthly = monthly_revenue(df)
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly.index, monthly.values, marker="o", color="#DD8452", linewidth=2)
        ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#DD8452")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Revenue (USD)", fontsize=12)
        ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: f"${y:,.0f}")
        )
        fig.tight_layout()
        _save_figure(fig, "monthly_revenue.png")

    # Rating distribution
    ratings = df["rating"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings, bins=20, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Rating (1–5)", fontsize=12)
    ax.set_ylabel("Number of Orders", fontsize=12)
    ax.set_title("Customer Rating Distribution", fontsize=14, fontweight="bold")
    ax.axvline(
        ratings.mean(), color="red", linestyle="--",
        linewidth=1.5, label=f"Mean: {ratings.mean():.2f}",
    )
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, "rating_distribution.png")

    # Top 5 countries by revenue (pie)
    country_rev = revenue_by_country(df).head(5)
    fig, ax = plt.subplots(figsize=(8, 8))
    _, _, autotexts = ax.pie(
        country_rev.values,
        labels=country_rev.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Set2.colors[: len(country_rev)],  # type: ignore[attr-defined]
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Top 5 Countries by Revenue", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, "country_revenue.png")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame) -> None:
    """Write the cleaned, enriched DataFrame to CSV and Parquet.

    Both formats are written so the BI team can use Parquet (which preserves
    dtypes and is faster for columnar queries) while CSV remains available
    for manual inspection.

    Args:
        df: Enriched DataFrame to save.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_path     = PROCESSED_DIR / "clean_smartphone_sales.csv"
    parquet_path = PROCESSED_DIR / "clean_smartphone_sales.parquet"

    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV     : %s (%d rows)", csv_path.name, len(df))

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved Parquet : %s (%d rows)", parquet_path.name, len(df))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(raw_path: pathlib.Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Run the full ETL pipeline end-to-end.

    Args:
        raw_path: Path to the raw CSV. Defaults to the project raw data path.

    Returns:
        Final enriched DataFrame (also written to disk).
    """
    logger.info("=" * 60)
    logger.info("Smartphone Sales Pipeline — starting")
    logger.info("=" * 60)

    df_raw      = load_data(raw_path)
    validate_raw(df_raw)
    df_clean    = clean_data(df_raw)
    df_enriched = add_derived_columns(df_clean)
    kpis        = compute_kpis(df_enriched)

    logger.info("-" * 60)
    logger.info("KPI SUMMARY")
    logger.info("-" * 60)
    logger.info("Total Orders       : %d",  kpis["total_orders"])
    logger.info("Total Revenue      : %s",  f"${kpis['total_revenue']:,.2f}")
    logger.info("Avg Order Value    : %s",  f"${kpis['average_order_value']:,.2f}")
    logger.info("Return Rate        : %.2f %%", kpis["return_rate_pct"])
    logger.info("Avg Rating         : %.2f",    kpis["average_rating"])
    logger.info("Unique Customers   : %d",  kpis["unique_customers"])
    logger.info("-" * 60)

    generate_charts(df_enriched)
    save_processed(df_enriched)

    logger.info("Pipeline complete.")
    return df_enriched


if __name__ == "__main__":
    try:
        run_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
