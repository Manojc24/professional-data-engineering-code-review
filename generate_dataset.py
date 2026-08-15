"""
generate_dataset.py
-------------------
Generates the synthetic smartphone_sales.csv dataset for the
Professional Data Engineering Code Review project.

This script is intentionally separate from the pipeline code.
It is a one-time utility used to produce the raw data file.

Run:
    python generate_dataset.py

Output:
    data/raw/smartphone_sales.csv  (~5,200 rows)

Design decisions:
- Fixed random seed (42) ensures the dataset is reproducible across machines.
- ~4% missing values injected at random positions in selected columns.
- ~80 exact duplicate rows appended at the end.
- Data quality issues are intentional — they exercise the pipeline's cleaning logic.
"""

import os
import random
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
N_CLEAN = 5_000          # clean base rows
N_DUPLICATES = 80        # intentional duplicate rows appended
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "data", "raw", "smartphone_sales.csv"
)

# ── Master reference data ─────────────────────────────────────────────────────

# Countries with intentional inconsistencies mixed in
CLEAN_COUNTRIES = [
    "India", "USA", "UK", "Germany", "France",
    "Japan", "Australia", "Canada", "Brazil", "Singapore",
]
DIRTY_COUNTRIES = [
    "india", "INDIA", "Inda", "india ",          # India variants
    "usa", "U.S.A", "United States",             # USA variants
    "uk", "U.K.", "United Kingdom",              # UK variants
    "germany", "GERMANY",                        # Germany variants
    "france", "FRANCE",                          # France variants
]

# Brands with intentional casing inconsistencies
CLEAN_BRANDS = ["Apple", "Samsung", "OnePlus", "Xiaomi", "Google", "Sony", "Motorola"]
DIRTY_BRANDS = [
    "apple", "APPLE",
    "samsung", "SAMSUNG",
    "oneplus", "ONEPLUS",
    "xiaomi", "XIAOMI",
    "google", "GOOGLE",
]

# Models per brand
MODELS = {
    "Apple":    ["iPhone 15", "iPhone 15 Pro", "iPhone 14", "iPhone SE"],
    "Samsung":  ["Galaxy S24", "Galaxy A54", "Galaxy Z Fold 5", "Galaxy S23"],
    "OnePlus":  ["OnePlus 12", "OnePlus Nord 3", "OnePlus 11"],
    "Xiaomi":   ["Xiaomi 14", "Redmi Note 13", "POCO F5", "Redmi 12"],
    "Google":   ["Pixel 8", "Pixel 8 Pro", "Pixel 7a"],
    "Sony":     ["Xperia 1 V", "Xperia 5 V", "Xperia 10 V"],
    "Motorola": ["Moto G84", "Edge 40 Pro", "Moto G53"],
}

# Canonical brand lookup (used when assigning models to dirty-brand rows)
BRAND_CANONICAL = {
    "apple": "Apple", "APPLE": "Apple",
    "samsung": "Samsung", "SAMSUNG": "Samsung",
    "oneplus": "OnePlus", "ONEPLUS": "OnePlus",
    "xiaomi": "Xiaomi", "XIAOMI": "Xiaomi",
    "google": "Google", "GOOGLE": "Google",
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "COD", "UPI", "Bank Transfer"]

# Date formats — mix to simulate real-world export inconsistencies
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_date(start_year: int = 2022, end_year: int = 2024) -> str:
    """Return a random date string in one of the mixed formats."""
    year  = rng.integers(start_year, end_year + 1)
    month = rng.integers(1, 13)
    # Keep day within valid range for month/year
    max_day = 28 if month == 2 else (30 if month in [4, 6, 9, 11] else 31)
    day = rng.integers(1, max_day + 1)
    fmt = random.choice(DATE_FORMATS)
    from datetime import date
    return date(year, month, day).strftime(fmt)


def random_brand_model(dirty: bool = False):
    """Return a (brand_str, model_str) tuple.

    If dirty=True, the brand string may have non-standard casing.
    The model is always looked up from the canonical brand.
    """
    if dirty and rng.random() < 0.4:
        brand_str = random.choice(DIRTY_BRANDS)
        canonical = BRAND_CANONICAL.get(brand_str, "Samsung")
    else:
        brand_str = random.choice(CLEAN_BRANDS)
        canonical = brand_str
    model_str = random.choice(MODELS[canonical])
    return brand_str, model_str


# ── Row generator ─────────────────────────────────────────────────────────────

def generate_rows(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        order_id = f"ORD-{100000 + i:06d}"
        customer_id = f"CUST-{rng.integers(10000, 99999):05d}"

        # --- Date ---
        date_str = random_date()
        # Inject a small number of malformed dates
        if rng.random() < 0.005:          # ~0.5 % of rows
            date_str = random.choice(["N/A", "31-02-2023", "0000-00-00", ""])

        # --- Country ---
        if rng.random() < 0.12:           # ~12 % dirty country names
            country = random.choice(DIRTY_COUNTRIES)
        else:
            country = random.choice(CLEAN_COUNTRIES)

        # --- Brand / Model ---
        brand, model = random_brand_model(dirty=True)

        # --- Price ---
        base_price = round(float(rng.uniform(149.99, 1499.99)), 2)
        r = rng.random()
        if r < 0.005:
            base_price = -999.0           # invalid: negative price
        elif r < 0.010:
            base_price = 0.0             # invalid: zero price
        elif r < 0.013:
            base_price = 99999.0         # outlier: extreme price

        # --- Quantity ---
        qty = int(rng.integers(1, 6))    # 1–5 normal
        r2 = rng.random()
        if r2 < 0.006:
            qty = -1                     # invalid: negative quantity
        elif r2 < 0.010:
            qty = 0                      # invalid: zero quantity
        elif r2 < 0.012:
            qty = 999                    # outlier: bulk order anomaly

        # --- Discount ---
        discount = round(float(rng.uniform(0.0, 0.40)), 3)
        r3 = rng.random()
        if r3 < 0.005:
            discount = round(float(rng.uniform(1.1, 2.0)), 3)   # invalid: > 1
        elif r3 < 0.008:
            discount = round(-float(rng.uniform(0.01, 0.10)), 3) # invalid: negative

        # --- Payment method ---
        payment = random.choice(PAYMENT_METHODS)

        # --- Rating ---
        rating = round(float(rng.uniform(1.0, 5.0)), 1)
        r4 = rng.random()
        if r4 < 0.005:
            rating = round(float(rng.uniform(5.1, 6.5)), 1)     # invalid: > 5
        elif r4 < 0.009:
            rating = 0.0                 # invalid: zero rating
        elif r4 < 0.012:
            rating = -1.0               # invalid: negative rating

        # --- Returned ---
        # Intentional inconsistency: mix of booleans, strings, and integers
        raw_returned = rng.random()
        if raw_returned < 0.10:
            returned = True
        elif raw_returned < 0.20:
            returned = False
        elif raw_returned < 0.28:
            returned = "yes"
        elif raw_returned < 0.35:
            returned = "no"
        elif raw_returned < 0.42:
            returned = "True"
        elif raw_returned < 0.50:
            returned = "False"
        elif raw_returned < 0.58:
            returned = 1
        elif raw_returned < 0.65:
            returned = 0
        elif raw_returned < 0.70:
            returned = "1"
        elif raw_returned < 0.75:
            returned = "0"
        else:
            # Will be set to NaN below by the missing-value injection step
            returned = False

        rows.append({
            "order_id":       order_id,
            "order_date":     date_str,
            "customer_id":    customer_id,
            "country":        country,
            "brand":          brand,
            "model":          model,
            "price":          base_price,
            "quantity":       qty,
            "discount":       discount,
            "payment_method": payment,
            "rating":         rating,
            "returned":       returned,
        })
    return rows


# ── Missing-value injection ───────────────────────────────────────────────────

def inject_missing(df: pd.DataFrame, missing_rate: float = 0.035) -> pd.DataFrame:
    """Randomly set ~3.5 % of values to NaN in selected columns.

    We deliberately leave order_id and customer_id intact because a pipeline
    should be able to identify rows even when other fields are missing.
    """
    nullable_cols = ["order_date", "country", "brand", "model",
                     "price", "quantity", "discount", "payment_method",
                     "rating", "returned"]
    df = df.copy()
    for col in nullable_cols:
        n_missing = int(len(df) * missing_rate)
        missing_idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[INFO] Generating {N_CLEAN:,} base rows …")
    rows = generate_rows(N_CLEAN)
    df = pd.DataFrame(rows)

    print(f"[INFO] Injecting missing values (~3.5 % per nullable column) …")
    df = inject_missing(df)

    # Append ~80 exact duplicates (simulates a double-export / ETL retry bug)
    print(f"[INFO] Appending {N_DUPLICATES} duplicate rows …")
    dup_idx = rng.choice(df.index, size=N_DUPLICATES, replace=False)
    duplicates = df.loc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle to distribute duplicates throughout the file
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[SUCCESS] Dataset saved to: {OUTPUT_PATH}")
    print(f"          Total rows  : {len(df):,}")
    print(f"          Total cols  : {len(df.columns)}")
    print(f"\n[SUMMARY] Missing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\n[SAMPLE]  First 3 rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
