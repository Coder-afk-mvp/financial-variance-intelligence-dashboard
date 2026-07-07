"""
Generates a synthetic, intentionally MESSY budget-vs-actual financial dataset.

Why messy on purpose?
Week 1 of the plan asks you to document data-quality issues (missing values,
duplicates, invalid values, inconsistent labels) BEFORE cleaning. Real company
budget data isn't public, so this script builds a dataset with the same kinds
of problems real finance data has, so the cleaning exercise is genuine rather
than pretend.

Run:
    python generate_data.py

Output:
    ../data/raw/financial_raw.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

DEPARTMENTS_CLEAN = ["Marketing", "Sales", "Operations", "IT", "HR"]
REGIONS_CLEAN = ["North America", "EMEA", "APAC"]
CATEGORIES_CLEAN = ["Payroll", "Travel", "Software", "Advertising", "Equipment"]

# messy label variants that map back to the clean ones above
DEPT_VARIANTS = {
    "Marketing": ["Marketing", "marketing", "MKT", "Marketing ", "Mktg"],
    "Sales": ["Sales", "sales", "SALES", " Sales"],
    "Operations": ["Operations", "Ops", "operations", "OPERATIONS"],
    "IT": ["IT", "I.T.", "it", "Information Technology"],
    "HR": ["HR", "H.R.", "hr", "Human Resources"],
}
REGION_VARIANTS = {
    "North America": ["North America", "N. America", "NA", "north america"],
    "EMEA": ["EMEA", "Emea", "emea"],
    "APAC": ["APAC", "Apac", "apac", "Asia Pacific"],
}
CATEGORY_VARIANTS = {
    "Payroll": ["Payroll", "payroll", "PAYROLL"],
    "Travel": ["Travel", "travel", "Travel & Expenses"],
    "Software": ["Software", "software", "SW"],
    "Advertising": ["Advertising", "advertising", "Ads"],
    "Equipment": ["Equipment", "equipment", "Equip."],
}

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %Y", "%d-%b-%Y"]

def messy_label(clean_value, variants_map):
    return random.choice(variants_map[clean_value])

def messy_date(dt):
    fmt = random.choice(DATE_FORMATS)
    if fmt == "%B %Y":
        return dt.strftime(fmt)  # e.g. "March 2024" (day info lost on purpose)
    return dt.strftime(fmt)

def build_base_rows():
    rows = []
    start = datetime(2024, 1, 1)
    months = [start + pd.DateOffset(months=i) for i in range(24)]  # 2 years

    txn_counter = 1
    for month in months:
        for dept in DEPARTMENTS_CLEAN:
            for region in REGIONS_CLEAN:
                for category in CATEGORIES_CLEAN:
                    base_budget = {
                        "Payroll": 80000, "Travel": 8000, "Software": 15000,
                        "Advertising": 20000, "Equipment": 10000
                    }[category]

                    # seasonal + random variation
                    seasonal = 1 + 0.15 * np.sin(2 * np.pi * month.month / 12)
                    noise = np.random.normal(1.0, 0.12)
                    budget = round(base_budget * seasonal, 2)
                    actual = round(budget * noise * np.random.normal(1.0, 0.08), 2)

                    rows.append({
                        "transaction_id": f"TXN-{txn_counter:06d}",
                        "date": month,
                        "department": dept,
                        "region": region,
                        "category": category,
                        "budget": budget,
                        "actual": actual,
                    })
                    txn_counter += 1
    return pd.DataFrame(rows)

def mess_it_up(df):
    df = df.copy()
    n = len(df)

    # 1. messy text labels
    df["department"] = df["department"].apply(lambda d: messy_label(d, DEPT_VARIANTS))
    df["region"] = df["region"].apply(lambda r: messy_label(r, REGION_VARIANTS))
    df["category"] = df["category"].apply(lambda c: messy_label(c, CATEGORY_VARIANTS))

    # 2. messy/mixed date formats (stored as strings, not datetime)
    df["date"] = df["date"].apply(messy_date)

    # cast to object dtype so we can mix floats, strings, and NaN in the same column
    df["actual"] = df["actual"].astype(object)
    df["budget"] = df["budget"].astype(object)

    # 3. missing values (~4-6% in a few columns)
    for col, frac in [("actual", 0.05), ("budget", 0.02), ("region", 0.03), ("category", 0.02)]:
        idx = df.sample(frac=frac, random_state=random.randint(1, 9999)).index
        df.loc[idx, col] = np.nan

    # 4. some amounts stored as text with currency symbols / commas
    idx = df.sample(frac=0.04, random_state=1).index
    df.loc[idx, "actual"] = df.loc[idx, "actual"].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else x
    )

    # 5. a few negative / impossible values (data entry errors)
    idx = df.sample(frac=0.01, random_state=2).index
    df.loc[idx, "actual"] = df.loc[idx, "actual"].apply(
        lambda x: -abs(float(str(x).replace("$", "").replace(",", ""))) if pd.notna(x) else x
    )

    # 6. a few extreme outliers (e.g. decimal/entry mistake, 50x too big)
    idx = df.sample(frac=0.005, random_state=3).index
    df.loc[idx, "actual"] = df.loc[idx, "actual"].apply(
        lambda x: float(str(x).replace("$", "").replace(",", "")) * 50 if pd.notna(x) else x
    )

    # 7. duplicate rows (~2%)
    dup_rows = df.sample(frac=0.02, random_state=4)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 8. shuffle so duplicates/issues aren't neatly grouped
    df = df.sample(frac=1, random_state=5).reset_index(drop=True)

    return df

if __name__ == "__main__":
    base = build_base_rows()
    messy = mess_it_up(base)
    out_path = "raw/financial_raw.csv"
    messy.to_csv(out_path, index=False)
    print(f"Wrote {len(messy)} rows to {out_path}")
    print("\nColumn dtypes as saved (before you clean them):")
    print(messy.dtypes)
