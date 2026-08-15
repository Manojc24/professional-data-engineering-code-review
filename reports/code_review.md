# Professional Code Review Report
## Project: Smartphone Sales Data Pipeline
## File Under Review: `junior_code/smartphone_pipeline.py`
## Reviewer: Senior Data Engineer
## Date: August 2026
## Review Standard: Google Engineering Code Review Guidelines + PEP 8

---

## Executive Summary

The junior engineer's pipeline attempts to load, clean, transform, and visualise
smartphone sales data. The **intent** of the code is correct: ingest a CSV, clean
data quality issues, calculate KPIs (revenue, return rate, brand rankings), generate
charts, and persist a clean output file.

However, the implementation contains **18 identified issues** — 5 critical, 4 high,
5 medium, and 4 low — that would produce **wrong business results** if run in
production, **crash** on any machine other than the author's, or make the code
**very difficult to maintain and test**.

The most serious problem is that revenue is calculated **before** the discount is
applied, meaning every revenue figure in every chart and print statement would be
**overstated**. The return-rate calculation is also incorrect, producing an inflated
figure whenever any `returned` values are NaN.

**No tests exist.** The pipeline cannot be safely deployed or modified without the
risk of silent regression.

---

## Issues Summary Table

| ID  | Severity | Category          | Location / Function       | Short Description                                  |
|-----|----------|-------------------|---------------------------|--------------------------------------------------- |
| C1  | CRITICAL  | Correctness       | `get_revenue()`           | Revenue ignores discount — wrong monetary figures  |
| C2  | CRITICAL  | Correctness       | `get_return_rate()`       | Denominator includes NaN rows — rate inflated      |
| C3  | CRITICAL  | Correctness       | `clean_data()`            | `dropna()` on full DataFrame removes valid rows    |
| C4  | CRITICAL  | Correctness       | `clean_data()`            | `pd.to_datetime()` crashes on malformed dates      |
| C5  | CRITICAL  | Correctness       | `main()`                  | `discount_amount` loop sets scalar, not per-row    |
| H1  | HIGH      | Maintainability   | `load_data()`             | Hard-coded absolute path — breaks on any machine   |
| H2  | HIGH      | Reliability       | `clean_data()`            | Chained assignment `df['brand'][i]` — data corruption risk |
| H3  | HIGH      | Reliability       | `clean_data()`            | Chained assignment `df['returned'][i]` — same risk |
| H4  | HIGH      | Design            | Entire file               | No input validation — silent failures on bad files |
| M1  | MEDIUM    | Performance       | `clean_data()` brand loop | `iterrows()` loop for string op — significantly slower than vectorised |
| M2  | MEDIUM    | Performance       | `clean_data()` return loop| Second `iterrows()` loop for boolean mapping       |
| M3  | MEDIUM    | Performance       | `clean_data()`            | `apply(lambda x: int(x))` — `astype(int)` is simpler & faster |
| M4  | MEDIUM    | Correctness       | `clean_data()`            | `rating > 0` keeps 0.1–0.9 — should be `>= 1.0`   |
| M5  | MEDIUM    | Maintainability   | `make_charts()`           | Charts saved to working directory, not output folder|
| L1  | LOW       | Naming / PEP 8    | Throughout                | Variable names `df2`, `top2`, `ctry2`, `x`, `val`  |
| L2  | LOW       | Documentation     | All functions             | Zero docstrings — no description of parameters or returns |
| L3  | LOW       | PEP 8             | Throughout                | Inconsistent spacing, missing blank lines, > 100 char lines |
| L4  | LOW       | Maintainability   | `main()`                  | Country re-normalisation duplicated at bottom of `main()` |

---

## Detailed Issue Analysis

---

### C1 — CRITICAL | Revenue ignores discount

**Location:** `get_revenue()` function, line ~34

**Code:**
```python
def get_revenue(df):
    df['revenue'] = df['price'] * df['quantity']
    return df
```

**Problem:**
Revenue is calculated as `price × quantity`, completely ignoring the `discount`
column. This means every downstream metric — total revenue, brand revenue, country
revenue, average order value, monthly trend chart — is **wrong**.

**Why it matters:**
If a customer bought a phone for $999 with a 20 % discount, the pipeline records
revenue as $999 when the actual realised revenue is $799.20. At 5,000 orders, the
cumulative overstatement could reach hundreds of thousands of dollars — directly
misleading business decisions.

**Fix:**
```python
def calculate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'revenue' column: price after discount multiplied by quantity."""
    df = df.copy()
    df["revenue"] = df["price"] * (1 - df["discount"]) * df["quantity"]
    return df
```

**Why the fix is better:**
Applies the discount correctly. Uses `.copy()` to avoid mutating the caller's
DataFrame. Renamed to `calculate_revenue` to be explicit about what is added.

---

### C2 — CRITICAL | Return-rate denominator is wrong

**Location:** `get_return_rate()`, line ~39

**Code:**
```python
def get_return_rate(df):
    x = df['returned'].sum()
    rate = x / len(df)
    return rate
```

**Problem:**
`len(df)` counts every row including those where `returned` is `NaN`.
`df['returned'].sum()` skips NaN values. The denominator is therefore larger than
the actual number of valid observations, making the rate **lower than reality**.

If 400 rows have NaN in `returned` out of 5,000 total, the computed rate is off
by ~8 % relative.

**Fix:**
```python
def calculate_return_rate(df: pd.DataFrame) -> float:
    """Return the fraction of orders that were returned.

    Only rows where 'returned' is non-null are considered.
    Avoids inflating the denominator with unknown/missing returns.
    """
    valid = df["returned"].dropna()
    if valid.empty:
        return 0.0
    return float(valid.sum() / len(valid))
```

---

### C3 — CRITICAL | `dropna()` on full DataFrame discards valid rows

**Location:** `clean_data()`, line ~16

**Code:**
```python
df.dropna(inplace=True)
```

**Problem:**
This single call removes **any row that has a NaN in ANY column**, including rows
that are otherwise completely valid. Since NaN values are injected across 10 of
12 columns at ~3.5 % rate, a row has a ≈ 30 % chance of being dropped even if
only one unimportant field (e.g. `payment_method`) is missing.

In a 5,000-row dataset this silently discards ~1,400–1,800 valid rows — a 28–36 %
data loss — without any warning.

**Fix:**
Handle missing values column-by-column based on the business rules for each field:
```python
# Drop rows where the key identifiers or measurable fields are missing
critical_cols = ["order_id", "price", "quantity", "returned"]
df = df.dropna(subset=critical_cols)

# Fill optional fields with sensible defaults or keep NaN for downstream use
df["discount"] = df["discount"].fillna(0.0)
df["payment_method"] = df["payment_method"].fillna("Unknown")
```

**Why the fix is better:**
Preserves rows that are valid for revenue and return-rate calculations even when
optional fields like `payment_method` or `rating` are missing.

---

### C4 — CRITICAL | `pd.to_datetime()` crashes on malformed dates

**Location:** `clean_data()`, line ~30

**Code:**
```python
df['order_date'] = pd.to_datetime(df['order_date'])
```

**Problem:**
Without `errors='coerce'`, any malformed date string like `"N/A"`, `"0000-00-00"`,
or an empty string causes a `ParserError` that **crashes the entire pipeline**,
meaning no data is processed at all.

**Fix:**
```python
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
# Log how many dates failed to parse — don't silently ignore
n_bad_dates = df["order_date"].isna().sum()
if n_bad_dates > 0:
    logger.warning("Could not parse %d order_date values — set to NaT", n_bad_dates)
```

---

### C5 — CRITICAL | `discount_amount` loop sets every row to the LAST row's value

**Location:** `main()`, near the bottom

**Code:**
```python
for i, row in df.iterrows():
    df['discount_amount'] = row['price'] * row['discount']
```

**Problem:**
Inside the loop, `df['discount_amount'] = scalar` broadcasts the **same scalar**
to the **entire column** on every iteration. After the loop completes, every single
row has the discount amount of the **last row** — not its own. This is a completely
wrong result and would go undetected without tests.

**Fix:**
```python
# Simple vectorised operation — no loop needed
df["discount_amount"] = df["price"] * df["discount"]
```

---

### H1 — HIGH | Hard-coded absolute file path

**Location:** `load_data()`, line ~12

**Code:**
```python
df = pd.read_csv("C:/Users/junior_dev/Desktop/projects/data/smartphone_sales.csv")
```

**Problem:**
This path is specific to one developer's machine. The code **cannot run** on any
other system — CI/CD, a colleague's machine, staging, or production — without
manual editing. This is a production anti-pattern.

**Fix:**
```python
import pathlib

# Resolve path relative to this file's location
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "smartphone_sales.csv"

def load_data(path: pathlib.Path = RAW_DATA_PATH) -> pd.DataFrame:
    ...
```

**Why the fix is better:**
Works on any machine regardless of directory structure. The default value means
callers don't need to pass the path every time. Tests can pass a different path
to a test fixture without modifying source code.

---

### H2 & H3 — HIGH | Chained assignment causes silent data corruption

**Location:** `clean_data()`, brand loop and returned loop

**Code:**
```python
df['brand'][i] = row['brand'].strip().title()     # H2
df['returned'][i] = True                           # H3
```

**Problem:**
`df['brand']` returns a view or a copy depending on internal Pandas state.
Writing back through `df['brand'][i] = ...` may silently **not update the
original DataFrame**, causing the loop to appear to work but producing wrong
results. Pandas raises `SettingWithCopyWarning` but this warning is easily
missed in a script.

**Fix:**
```python
df["brand"] = df["brand"].str.strip().str.title()          # H2 — vectorised, safe
df["returned"] = df["returned"].map(RETURN_VALUE_MAP)       # H3 — explicit map
```

---

### H4 — HIGH | No input validation

**Location:** `load_data()` — the function has no validation at all.

**Problem:**
If the file does not exist, `pd.read_csv()` raises a `FileNotFoundError` with a
cryptic traceback. If the schema changes (a column is renamed upstream), the
pipeline silently produces wrong results or crashes mid-run with a `KeyError`.

In a production pipeline that runs on a schedule, this means hours of data loss
before anyone notices.

**Fix:**
```python
REQUIRED_COLUMNS = {
    "order_id", "order_date", "customer_id", "country",
    "brand", "model", "price", "quantity", "discount",
    "payment_method", "rating", "returned",
}

def load_data(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input file is missing required columns: {missing_cols}")
    return df
```

---

### M1 — MEDIUM | `iterrows()` loop for brand name cleaning

**Location:** `clean_data()`, brand loop

**Code:**
```python
for i, row in df.iterrows():
    df['brand'][i] = row['brand'].strip().title()
```

**Problem:**
`iterrows()` is the slowest possible way to apply a string operation in Pandas.
For 5,000 rows it takes milliseconds, but for 500,000 rows (a realistic daily
load) this becomes seconds to minutes. More importantly, it establishes a pattern
that junior engineers copy throughout the codebase.

**Benchmark context:** `.str.title()` operates in C-compiled vectorised code.
`iterrows()` is pure Python with row-by-row object overhead.

**Fix:**
```python
df["brand"] = df["brand"].str.strip().str.title()
```

One line. Significantly faster in practice (avoids Python-level row iteration). Safer (no chained assignment).

---

### M2 — MEDIUM | Second `iterrows()` loop for boolean mapping

**Location:** `clean_data()`, returned loop

**Code:**
```python
for i, row in df.iterrows():
    val = row['returned']
    if val == 'yes' or val == 'True' or val == '1' or val == 1 or val == True:
        df['returned'][i] = True
    else:
        df['returned'][i] = False
```

**Problem:**
Second `iterrows()` loop. Same performance issue as M1. Additionally, the `else`
branch maps NaN to `False`, which is semantically wrong — we cannot know whether
a NaN-returned order was or wasn't returned.

**Fix:**
```python
TRUTHY_RETURN_VALUES = {"yes", "true", "1", 1, True}

def normalise_returned(value) -> bool | None:
    if pd.isna(value):
        return None           # preserve unknown — do not assume not-returned
    return str(value).strip().lower() in {"yes", "true", "1"}

df["returned"] = df["returned"].map(normalise_returned)
```

---

### M3 — MEDIUM | Unnecessary `apply(lambda)` for type cast

**Location:** `clean_data()`, quantity

**Code:**
```python
df['quantity'] = df['quantity'].apply(lambda x: int(x))
```

**Problem:**
`apply()` with a lambda is Python-level iteration. `astype()` calls the C-level
numpy type conversion and is significantly faster. The `lambda` adds no clarity.

**Fix:**
```python
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
```

Using `Int64` (nullable integer) preserves NaN instead of crashing.

---

### M4 — MEDIUM | Incorrect rating boundary condition

**Location:** `clean_data()`

**Code:**
```python
df = df[df['rating'] > 0]
df = df[df['rating'] < 6]
```

**Problem:**
`rating > 0` allows values like `0.1`, `0.5`, `0.9` — which are not valid on a
1–5 scale. The valid range should be `1.0 ≤ rating ≤ 5.0`.

**Fix:**
```python
MIN_RATING, MAX_RATING = 1.0, 5.0
df = df[df["rating"].between(MIN_RATING, MAX_RATING)]
```

---

### M5 — MEDIUM | Charts saved to working directory

**Location:** `make_charts()`

**Code:**
```python
plt.savefig('brand_revenue.png')
```

**Problem:**
Saves to wherever Python is executed from, not a predictable project folder. On
CI or production this pollutes the execution directory. Charts are also not
tracked properly in version control.

**Fix:**
```python
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(CHARTS_DIR / "brand_revenue.png", dpi=150, bbox_inches="tight")
```

---

### L1 — LOW | Poor variable names

**Locations:** Throughout the file

| Bad name | Context | Better name |
|---|---|---|
| `df2` | Result of `get_revenue` | `df_with_revenue` |
| `top2` | Sorted brand revenue | `brand_revenue_sorted` |
| `ctry2` | Sorted country revenue | `country_revenue_sorted` |
| `x` | Sum of returned values | `total_returned` |
| `val` | Current cell value | `returned_value` |

**Why it matters:** Code is read far more often than it is written. Poor names
force every future reader (including the author, 3 months later) to trace
execution to understand what `df2` is.

---

### L2 — LOW | Missing docstrings on all functions

**Locations:** All 6 functions

**Problem:**
Not one function has a docstring. A reader cannot tell what parameters are
expected, what types they should be, what the function returns, or what
side effects it has.

**Fix example:**
```python
def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load the raw smartphone sales CSV file.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        Raw DataFrame with all original columns preserved.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
```

---

### L3 — LOW | PEP 8 violations

**Locations:** Throughout

| Violation | Example |
|---|---|
| Missing blank line after function | `return df` immediately followed by `def` |
| Inconsistent spacing | `df['country']=df['country'].str.strip()` |
| Comments inline instead of above | Style inconsistency |
| Missing space after `#` | `#load data` instead of `# load data` |

---

### L4 — LOW | Country normalisation duplicated in `main()`

**Location:** Bottom of `main()`

**Code:**
```python
# fix country names again because some are still lowercase after merge
df['country'] = df['country'].str.strip()
df['country'] = df['country'].str.lower()
```

**Problem:**
This is copy-pasted from `clean_data()`. If the normalisation logic ever
changes (e.g. to use `.str.title()` instead of `.str.lower()`), it must be
updated in two places — and someone will inevitably miss one.

This is a classic DRY (Don't Repeat Yourself) violation.

**Fix:**
Extract into a `normalise_country()` helper function and call it once.

---

## Summary of Recommendations

### Must Fix Before Merge (CRITICAL + HIGH)
1. Apply discount in revenue calculation.
2. Fix return-rate denominator.
3. Replace `dropna()` with column-specific handling.
4. Add `errors='coerce'` to date parsing.
5. Fix `discount_amount` vectorisation bug.
6. Remove hard-coded path; use relative path from `__file__`.
7. Replace chained assignments with `.str` methods.
8. Add schema and file-existence validation.

### Should Fix (MEDIUM)
9. Replace both `iterrows()` loops with vectorised operations.
10. Use `astype()` or `pd.to_numeric()` instead of `apply(lambda)`.
11. Fix rating boundary to `>= 1.0` and `<= 5.0`.
12. Save charts to a predictable project directory.

### Nice to Have (LOW)
13. Rename opaque variables.
14. Add docstrings to all functions.
15. Fix PEP 8 formatting.
16. Deduplicate country normalisation.

### Missing Entirely
17. **No tests.** A pipeline of this complexity requires at least unit tests for
    revenue calculation, return rate, date parsing, and data validation.
18. **No logging.** `print()` statements are not appropriate for a production
    pipeline. Use Python's `logging` module so log level can be controlled.

---

*Review completed in accordance with Google Engineering code review principles:
correctness first, then design, readability, and style.*
