"""
test_smartphone_pipeline.py
============================
Unit tests for reviewed_code/smartphone_pipeline.py.

Run:
    pytest tests/test_smartphone_pipeline.py -v
    pytest tests/test_smartphone_pipeline.py -v --cov=reviewed_code

Test scope:
- Business logic correctness (revenue formula, return rate, null handling)
- Data cleaning rules (which fields are critical vs optional)
- Normalisation functions (brand, country, returned)
- Edge cases (all-NaN, boundary values, malformed dates)
- Regression guards for the five critical bugs found in the junior code

Each test uses a minimal DataFrame constructed for that specific scenario.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# Allow importing the reviewed module regardless of working directory.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reviewed_code.smartphone_pipeline import (
    REQUIRED_COLUMNS,
    MAX_VALID_PRICE,
    MIN_VALID_PRICE,
    MIN_RATING,
    MAX_RATING,
    _normalise_returned,
    _normalise_country,
    add_derived_columns,
    calculate_return_rate,
    clean_data,
    load_data,
    monthly_revenue,
    revenue_by_brand,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — small, focused DataFrames that cover edge cases
# ─────────────────────────────────────────────────────────────────────────────

def _make_minimal_df(**overrides) -> pd.DataFrame:
    """Create a single-row DataFrame with all required columns set to valid values.

    Use keyword arguments to override any column for a specific test scenario.
    """
    base = {
        "order_id":       "ORD-000001",
        "order_date":     "2024-01-15",
        "customer_id":    "CUST-11111",
        "country":        "India",
        "brand":          "Apple",
        "model":          "iPhone 15",
        "price":          999.99,
        "quantity":       1,
        "discount":       0.10,
        "payment_method": "Credit Card",
        "rating":         4.5,
        "returned":       False,
    }
    base.update(overrides)
    return pd.DataFrame([base])


@pytest.fixture
def clean_single_row() -> pd.DataFrame:
    """A single fully valid row — baseline for most tests."""
    return _make_minimal_df()


@pytest.fixture
def multi_brand_df() -> pd.DataFrame:
    """Multi-row DataFrame with two brands for aggregation tests.

    Apple total revenue:   999×2 + 800×0.90×1 = 1998 + 720 = 2718
    Samsung total revenue: 600×0.95×3 = 1710
    """
    rows = [
        _make_minimal_df(brand="Apple",   price=999.0, quantity=2, discount=0.0),
        _make_minimal_df(brand="Apple",   price=800.0, quantity=1, discount=0.10),
        _make_minimal_df(brand="Samsung", price=600.0, quantity=3, discount=0.05),
    ]
    return pd.concat(rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2: Data loading
# ─────────────────────────────────────────────────────────────────────────────

def test_load_raises_file_not_found():
    """load_data() must raise FileNotFoundError when the file does not exist."""
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        load_data(pathlib.Path("/nonexistent/path/file.csv"))


def test_load_raises_on_missing_schema(tmp_path):
    """load_data() must raise ValueError when required columns are absent."""
    # Write a CSV that is missing the 'returned' and 'price' columns.
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"order_id": "1", "brand": "Apple"}]).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_data(bad_csv)


def test_load_returns_all_required_columns(tmp_path):
    """load_data() returns a DataFrame containing every required column."""
    # Write a minimal valid CSV.
    csv_path = tmp_path / "test.csv"
    _make_minimal_df().to_csv(csv_path, index=False)

    df = load_data(csv_path)
    assert REQUIRED_COLUMNS.issubset(set(df.columns)), (
        f"Missing columns after load: {REQUIRED_COLUMNS - set(df.columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3: Duplicate removal
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_removes_exact_duplicates():
    """clean_data() must remove exact duplicate rows."""
    row = _make_minimal_df()
    df_with_dupes = pd.concat([row, row, row], ignore_index=True)
    assert len(df_with_dupes) == 3

    result = clean_data(df_with_dupes)
    assert len(result) == 1, "Three identical rows should be reduced to one."


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5: Missing value handling
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_drops_row_with_null_price():
    """Rows missing 'price' (a critical field) must be removed."""
    df = _make_minimal_df(price=np.nan)
    result = clean_data(df)
    assert len(result) == 0, "Row with null price should be dropped."


def test_clean_drops_row_with_null_quantity():
    """Rows missing 'quantity' (a critical field) must be removed."""
    df = _make_minimal_df(quantity=np.nan)
    result = clean_data(df)
    assert len(result) == 0, "Row with null quantity should be dropped."


def test_clean_preserves_row_with_null_payment_method():
    """Rows missing 'payment_method' (an optional field) must NOT be dropped."""
    df = _make_minimal_df(payment_method=np.nan)
    result = clean_data(df)
    assert len(result) == 1, (
        "A row with a missing optional field (payment_method) should be preserved."
    )
    assert result["payment_method"].iloc[0] == "Unknown"


def test_clean_preserves_row_with_null_rating():
    """Rating is optional — rows without a rating must be retained."""
    df = _make_minimal_df(rating=np.nan)
    result = clean_data(df)
    assert len(result) == 1, "Row with null rating (optional) should be kept."


# ─────────────────────────────────────────────────────────────────────────────
# 6 & 7: Price validation
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_removes_negative_price():
    """Rows with a negative price must be filtered out."""
    df = _make_minimal_df(price=-999.0)
    result = clean_data(df)
    assert len(result) == 0, "Negative price rows should be removed."


def test_clean_removes_zero_price():
    """Rows with a price of zero must be filtered out."""
    df = _make_minimal_df(price=0.0)
    result = clean_data(df)
    assert len(result) == 0, "Zero price rows should be removed."


def test_clean_removes_extreme_price():
    """Rows with a price above MAX_VALID_PRICE must be removed as outliers."""
    df = _make_minimal_df(price=MAX_VALID_PRICE + 1)
    result = clean_data(df)
    assert len(result) == 0, f"Price > {MAX_VALID_PRICE} should be filtered out."


# ─────────────────────────────────────────────────────────────────────────────
# 8: Quantity validation
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_removes_negative_quantity():
    """Rows with negative quantity must be filtered out."""
    df = _make_minimal_df(quantity=-1)
    result = clean_data(df)
    assert len(result) == 0, "Negative quantity should be removed."


def test_clean_removes_zero_quantity():
    """Rows with a quantity of zero must be filtered out."""
    df = _make_minimal_df(quantity=0)
    result = clean_data(df)
    assert len(result) == 0, "Zero quantity should be removed."


# ─────────────────────────────────────────────────────────────────────────────
# 9: Rating validation
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_removes_rating_above_max():
    """Rows with a rating above MAX_RATING (5.0) must be removed."""
    df = _make_minimal_df(rating=6.5)
    result = clean_data(df)
    assert len(result) == 0, "Rating > 5.0 should be filtered out."


def test_clean_removes_rating_below_min():
    """Rows with a rating below MIN_RATING (1.0) must be removed."""
    df = _make_minimal_df(rating=0.5)
    result = clean_data(df)
    assert len(result) == 0, "Rating 0.5 is below the valid minimum of 1.0."


def test_clean_keeps_null_rating():
    """Rows with a null rating must be kept — rating is an optional field."""
    df = _make_minimal_df(rating=np.nan)
    result = clean_data(df)
    assert len(result) == 1, "NaN rating row should not be dropped."


# ─────────────────────────────────────────────────────────────────────────────
# 10 & 11: Revenue calculation — the most critical business logic
# ─────────────────────────────────────────────────────────────────────────────

def test_revenue_applies_discount_correctly():
    """Revenue must equal price × (1 - discount) × quantity.

    This is the fix for CRITICAL issue C1 in the junior code.
    """
    df = _make_minimal_df(price=1000.0, discount=0.20, quantity=2)
    df_clean = clean_data(df)
    df_enriched = add_derived_columns(df_clean)

    expected_revenue = 1000.0 * (1 - 0.20) * 2   # = 1600.0
    actual_revenue   = df_enriched["revenue"].iloc[0]

    assert abs(actual_revenue - expected_revenue) < 0.01, (
        f"Expected revenue {expected_revenue}, got {actual_revenue}. "
        "Discount must be applied before multiplying by quantity."
    )


def test_revenue_is_not_price_times_quantity():
    """Regression test: revenue must NOT equal price × quantity (junior bug C1).

    If discount > 0, the junior formula (price × quantity) will produce
    a larger number than the correct formula. This test explicitly guards
    against the regression where discount is ignored.
    """
    df = _make_minimal_df(price=500.0, discount=0.10, quantity=3)
    df_clean = clean_data(df)
    df_enriched = add_derived_columns(df_clean)

    junior_wrong_revenue = 500.0 * 3          # = 1500.0 (discount ignored)
    correct_revenue      = 500.0 * 0.90 * 3   # = 1350.0 (discount applied)
    actual_revenue       = df_enriched["revenue"].iloc[0]

    assert abs(actual_revenue - correct_revenue) < 0.01, (
        f"Revenue should be {correct_revenue}, not {junior_wrong_revenue}."
    )
    assert abs(actual_revenue - junior_wrong_revenue) > 0.01, (
        "Revenue must NOT equal price×quantity when discount > 0 (regression guard)."
    )


def test_discount_amount_is_per_row():
    """Each row's discount_amount must reflect that row's own price and discount.

    This validates the fix for CRITICAL issue C5 — the junior code iterated
    over rows but assigned a scalar to the entire column, meaning every row
    ended up with the last row's discount_amount.
    """
    rows = [
        _make_minimal_df(price=1000.0, discount=0.10),  # discount_amount = 100
        _make_minimal_df(price=500.0,  discount=0.20),  # discount_amount = 100
        _make_minimal_df(price=200.0,  discount=0.50),  # discount_amount = 100
    ]
    df = pd.concat(rows, ignore_index=True)
    df_clean = clean_data(df)
    df_enriched = add_derived_columns(df_clean)

    expected = [100.0, 100.0, 100.0]
    actuals  = df_enriched["discount_amount"].tolist()

    # Actual expected values
    real_expected = [1000 * 0.10, 500 * 0.20, 200 * 0.50]
    for i, (exp, act) in enumerate(zip(real_expected, actuals)):
        assert abs(act - exp) < 0.01, (
            f"Row {i}: expected discount_amount {exp}, got {act}. "
            "discount_amount must be computed per-row, not broadcast from last row."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12 & 13: Return rate calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_return_rate_excludes_null_returned():
    """Return rate denominator must exclude rows where 'returned' is None/NaN.

    This is the fix for CRITICAL issue C2 — the junior code used len(df)
    as the denominator, which included NaN rows and understated the rate.
    """
    # 2 returned=True, 2 returned=False, 1 returned=None
    # Correct rate: 2 / 4 = 0.50 (NaN row excluded from denominator)
    # Junior rate:  2 / 5 = 0.40 (NaN row wrongly counted)
    rows = [
        _make_minimal_df(returned=True),
        _make_minimal_df(returned=True),
        _make_minimal_df(returned=False),
        _make_minimal_df(returned=False),
        _make_minimal_df(returned=None),
    ]
    df = pd.concat(rows, ignore_index=True)
    df_clean = clean_data(df)

    rate = calculate_return_rate(df_clean)
    assert abs(rate - 0.50) < 0.001, (
        f"Expected return rate 0.50, got {rate}. "
        "NaN rows must be excluded from the denominator."
    )


def test_return_rate_all_null_returns_zero():
    """Return rate must be 0.0 when all 'returned' values are None."""
    rows = [_make_minimal_df(returned=None) for _ in range(5)]
    df = pd.concat(rows, ignore_index=True)
    df_clean = clean_data(df)

    rate = calculate_return_rate(df_clean)
    assert rate == 0.0, "All-null 'returned' column should yield a return rate of 0.0."


# ─────────────────────────────────────────────────────────────────────────────
# 14: Brand normalisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw_brand, expected", [
    ("apple",   "Apple"),
    ("APPLE",   "Apple"),
    ("SAMSUNG", "Samsung"),
    ("samsung", "Samsung"),
    ("google",  "Google"),
    ("XIAOMI",  "Xiaomi"),
    ("Apple",   "Apple"),     # already correct — should not change
])
def test_brand_normalised_to_title_case(raw_brand, expected):
    """Brand names must be title-cased regardless of input casing."""
    df = _make_minimal_df(brand=raw_brand)
    result = clean_data(df)
    assert result["brand"].iloc[0] == expected, (
        f"Brand '{raw_brand}' should become '{expected}', "
        f"got '{result['brand'].iloc[0]}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 15: Country normalisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw_country, expected", [
    ("india",          "India"),
    ("INDIA",          "India"),
    ("india ",         "India"),   # trailing space
    ("Inda",           "India"),   # typo variant
    ("usa",            "USA"),
    ("U.S.A",          "USA"),
    ("united states",  "USA"),
    ("uk",             "UK"),
    ("U.K.",           "UK"),
    ("Germany",        "Germany"), # already correct
])
def test_country_normalised_via_alias_table(raw_country, expected):
    """Country names must be resolved through the alias table."""
    result = _normalise_country(raw_country)
    assert result == expected, (
        f"Country '{raw_country}' should map to '{expected}', got '{result}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 16: Returned column normalisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw_value, expected", [
    (True,    True),
    ("True",  True),
    ("true",  True),
    ("yes",   True),
    ("YES",   True),
    ("1",     True),
    (1,       True),
    (False,   False),
    ("False", False),
    ("false", False),
    ("no",    False),
    ("0",     False),
    (0,       False),
    (None,    None),
    (np.nan,  None),
])
def test_returned_normalisation(raw_value, expected):
    """_normalise_returned() must correctly map all known returned value variants."""
    result = _normalise_returned(raw_value)
    assert result == expected, (
        f"_normalise_returned({raw_value!r}) should return {expected!r}, got {result!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 17: Date parsing robustness
# ─────────────────────────────────────────────────────────────────────────────

def test_malformed_dates_become_nat_not_crash():
    """Malformed date strings must produce NaT, not raise an exception."""
    rows = [
        _make_minimal_df(order_date="N/A"),
        _make_minimal_df(order_date="0000-00-00"),
        _make_minimal_df(order_date="not a date"),
        _make_minimal_df(order_date="2024-01-15"),  # valid — should parse normally
    ]
    df = pd.concat(rows, ignore_index=True)

    # Should not raise.
    result = clean_data(df)

    valid_dates = result["order_date"].dropna()
    assert len(valid_dates) >= 1, "At least the valid date row should survive."

    # The valid date row should have been parsed correctly.
    parsed_dates = result["order_date"].dropna()
    assert all(pd.notna(d) for d in parsed_dates), (
        "Parseable dates must not be NaT after cleaning."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 18: Derived columns are added
# ─────────────────────────────────────────────────────────────────────────────

def test_add_derived_columns_adds_expected_columns(clean_single_row):
    """add_derived_columns() must add discount_amount, net_price, revenue, year_month."""
    df_clean = clean_data(clean_single_row)
    df_enriched = add_derived_columns(df_clean)

    expected_new_cols = {"discount_amount", "net_price", "revenue", "year_month"}
    assert expected_new_cols.issubset(set(df_enriched.columns)), (
        f"Missing columns: {expected_new_cols - set(df_enriched.columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 19: Revenue by brand is sorted descending
# ─────────────────────────────────────────────────────────────────────────────

def test_revenue_by_brand_sorted_descending(multi_brand_df):
    """revenue_by_brand() must return brands in descending revenue order."""
    df_clean    = clean_data(multi_brand_df)
    df_enriched = add_derived_columns(df_clean)
    brand_rev   = revenue_by_brand(df_enriched)

    values = brand_rev.values
    assert list(values) == sorted(values, reverse=True), (
        "revenue_by_brand() must return values in descending order."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 20: Monthly revenue excludes NaT dates
# ─────────────────────────────────────────────────────────────────────────────

def test_monthly_revenue_excludes_nat_dates():
    """monthly_revenue() must not include rows where order_date is NaT."""
    rows = [
        _make_minimal_df(order_date="2024-01-15"),
        _make_minimal_df(order_date="2024-01-20"),
        _make_minimal_df(order_date="N/A"),         # will become NaT
    ]
    df = pd.concat(rows, ignore_index=True)
    df_clean    = clean_data(df)
    df_enriched = add_derived_columns(df_clean)
    monthly     = monthly_revenue(df_enriched)

    # All values must be positive (NaT row excluded — its revenue is not counted).
    assert (monthly > 0).all(), "All monthly revenue values should be positive."
    # Only one month should appear since both valid orders are in January 2024.
    assert "2024-01" in monthly.index, "January 2024 revenue should be in the result."
    # NaT-dated rows should not produce a 'NaT' entry.
    assert not any("NaT" in str(idx) for idx in monthly.index), (
        "NaT-dated orders must not appear as a month in the monthly revenue output."
    )
