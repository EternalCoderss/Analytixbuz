"""
Deterministic demo dataset generator for Analytixbuz.
Generates 3 cross-industry datasets with controlled noise, missing values,
duplicates, and outliers to demonstrate the Data Quality Engine.

Usage:
    python data/generate_demo_datasets.py
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 1000
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _inject_noise(df: pd.DataFrame, rng: np.random.Generator,
                  missing_pct: float = 0.02, dup_pct: float = 0.01,
                  outlier_pct: float = 0.015) -> pd.DataFrame:
    """Inject missing values, duplicates, and outliers into a DataFrame."""
    n, m = df.shape
    target_col_idx = m - 1  # Last column is always the target

    # --- Inject missing values (~2%) --- skip target column
    num_missing = int(n * m * missing_pct)
    for _ in range(num_missing):
        r = rng.integers(0, n)
        c = rng.integers(0, target_col_idx)  # Exclude target column
        df.iat[r, c] = np.nan

    # --- Inject duplicate rows (~1%) ---
    num_dups = int(n * dup_pct)
    if num_dups > 0:
        dup_indices = rng.integers(0, n, size=num_dups)
        dup_rows = df.iloc[dup_indices].copy()
        df = pd.concat([df, dup_rows], ignore_index=True)

    # --- Inject outliers (~1.5%) in numeric columns (skip target & ID columns) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col_name = df.columns[target_col_idx]
    numeric_cols = [c for c in numeric_cols if c != target_col_name]
    num_outliers = int(n * outlier_pct)
    for col in numeric_cols:
        col_std = df[col].std()
        col_mean = df[col].mean()
        if pd.isna(col_std) or col_std == 0:
            continue
        outlier_indices = rng.integers(0, len(df), size=max(1, num_outliers // len(numeric_cols)))
        for idx in outlier_indices:
            direction = rng.choice([-1, 1])
            df.at[idx, col] = col_mean + direction * rng.uniform(4, 6) * col_std

    return df


def generate_machine_data(rng: np.random.Generator) -> pd.DataFrame:
    """
    Manufacturing: Predictive Maintenance dataset.
    failure is driven by vibration, temperature, pressure, machine_age with interactions.
    """
    n = N_ROWS
    machine_id = [f"M-{i:04d}" for i in range(n)]
    temperature = rng.normal(70, 12, n)
    pressure = rng.normal(30, 5, n)
    vibration = rng.exponential(2.5, n) + rng.normal(0, 0.5, n)
    speed = rng.normal(1500, 200, n)
    machine_age = rng.uniform(0.5, 15, n)

    # Ground-truth: failure driven by vibration, temperature, pressure, age
    logit = (
        -3.5
        + 0.50 * (vibration - 2.5)
        + 0.08 * (temperature - 70)
        + 0.10 * (pressure - 30)
        + 0.15 * (machine_age - 7)
        + 0.04 * vibration * temperature / 10.0
        + 0.05 * vibration * machine_age / 5.0
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    failure = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "machine_id": machine_id,
        "temperature": np.round(temperature, 2),
        "pressure": np.round(pressure, 2),
        "vibration": np.round(vibration, 3),
        "speed": np.round(speed, 1),
        "machine_age": np.round(machine_age, 1),
        "failure": failure,
    })
    return _inject_noise(df, rng)


def generate_retail_demand(rng: np.random.Generator) -> pd.DataFrame:
    """
    Retail: Demand Forecasting / Regression dataset.
    demand is driven by discount, price, sales, inventory, and seasonal date patterns.
    """
    n = N_ROWS
    start_date = pd.Timestamp("2023-01-01")
    dates = [start_date + pd.Timedelta(days=int(i)) for i in rng.integers(0, 730, n)]
    dates.sort()

    products = rng.choice(["Electronics", "Clothing", "Grocery", "Furniture", "Sports"], n)
    price = rng.uniform(10, 500, n)
    discount = rng.uniform(0, 0.4, n)
    sales = rng.poisson(50, n).astype(float)
    inventory = rng.integers(10, 500, n).astype(float)

    # Seasonal component from date
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    weekday = np.array([d.weekday() for d in dates])
    seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365) + 5 * (weekday < 5).astype(float)

    # Ground-truth: demand driven by discount, price, sales, inventory, seasonal
    demand = (
        50.0
        + 150.0 * discount
        - 0.12 * price
        + 0.8 * sales
        - 0.08 * inventory
        + seasonal
        + rng.normal(0, 3, n)
    )
    demand = np.maximum(demand, 0).round(1)

    df = pd.DataFrame({
        "date": dates,
        "product": products,
        "price": np.round(price, 2),
        "discount": np.round(discount, 3),
        "sales": sales,
        "inventory": inventory,
        "demand": demand,
    })
    return _inject_noise(df, rng)


def generate_customer_churn(rng: np.random.Generator) -> pd.DataFrame:
    """
    Customer Analytics: Churn & Risk Analysis dataset.
    churn is driven by support_calls, contract type, tenure, spending.
    """
    n = N_ROWS
    customer_id = [f"CUST-{i:05d}" for i in range(n)]
    age = rng.integers(18, 75, n).astype(float)
    tenure = rng.exponential(24, n).round(1)
    spending = rng.lognormal(4.5, 0.8, n).round(2)
    support_calls = rng.poisson(2, n).astype(float)
    contracts = rng.choice(["Month-to-Month", "One-Year", "Two-Year"], n, p=[0.45, 0.30, 0.25])

    # Ground-truth: churn driven by support_calls, contract, tenure, spending
    contract_score = np.where(contracts == "Month-to-Month", 1.5, np.where(contracts == "One-Year", 0.0, -1.0))

    logit = (
        -2.5
        + 0.45 * (support_calls - 2)
        + contract_score
        - 0.04 * (tenure - 24)
        + 0.002 * (spending - 100)
        - 0.01 * (age - 40)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "tenure": tenure,
        "spending": np.round(spending, 2),
        "support_calls": support_calls,
        "contract": contracts,
        "churn": churn,
    })
    return _inject_noise(df, rng)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating Manufacturing: machine_data.csv ...")
    machine_df = generate_machine_data(rng)
    machine_df.to_csv(os.path.join(DATA_DIR, "machine_data.csv"), index=False)
    print(f"  -> {len(machine_df)} rows, {len(machine_df.columns)} cols, "
          f"failure rate: {machine_df['failure'].mean():.2%}")

    print("Generating Retail: retail_demand.csv ...")
    retail_df = generate_retail_demand(rng)
    retail_df.to_csv(os.path.join(DATA_DIR, "retail_demand.csv"), index=False)
    print(f"  -> {len(retail_df)} rows, {len(retail_df.columns)} cols, "
          f"demand mean: {retail_df['demand'].mean():.1f}")

    print("Generating Customer Analytics: customer_churn.csv ...")
    churn_df = generate_customer_churn(rng)
    churn_df.to_csv(os.path.join(DATA_DIR, "customer_churn.csv"), index=False)
    print(f"  -> {len(churn_df)} rows, {len(churn_df.columns)} cols, "
          f"churn rate: {churn_df['churn'].mean():.2%}")

    print("\n✅ All 3 demo datasets generated successfully!")


if __name__ == "__main__":
    main()
