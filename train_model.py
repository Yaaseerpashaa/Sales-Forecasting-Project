# ============================================================
# train_model.py
# This file:
#   1. Loads the sales dataset
#   2. Cleans and prepares the data
#   3. Creates useful features from dates
#   4. Trains 3 ML models and compares them
#   5. Saves the best model as a .pkl file
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Use non-GUI backend so plots save to files
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, joblib

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from xgboost                 import XGBRegressor

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1 – LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 55)
print("  SALES FORECASTING – MODEL TRAINING")
print("=" * 55)

DATA_PATH = os.path.join("dataset", "sales.csv")

if not os.path.exists(DATA_PATH):
    print("\n[INFO] No dataset found – generating a realistic sample dataset…")

    np.random.seed(42)
    dates     = pd.date_range(start="2020-01-01", end="2023-12-31", freq="W")
    n         = len(dates)

    # Simulate realistic sales with trend + seasonality + noise
    trend       = np.linspace(1000, 3000, n)
    seasonality = 500 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise       = np.random.normal(0, 200, n)

    df = pd.DataFrame({
        "Date"       : dates,
        "Store"      : np.random.choice([1, 2, 3, 4, 5], n),
        "Department" : np.random.choice(["Electronics", "Clothing", "Grocery", "Toys"], n),
        "Sales"      : np.maximum(0, trend + seasonality + noise).round(2),
        "Promotion"  : np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "Holiday"    : np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "Temperature": np.random.uniform(15, 40, n).round(1),
        "Fuel_Price" : np.random.uniform(2.5, 4.5, n).round(2),
        "CPI"        : np.random.uniform(200, 230, n).round(2),
        "Unemployment": np.random.uniform(5, 10, n).round(2),
    })

    os.makedirs("dataset", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[OK] Sample dataset saved → {DATA_PATH}  ({n} rows)\n")
else:
    df = pd.read_csv(DATA_PATH)
    print(f"\n[OK] Dataset loaded → {DATA_PATH}  ({df.shape[0]} rows, {df.shape[1]} cols)\n")

print("First 5 rows:")
print(df.head(), "\n")

# ─────────────────────────────────────────────
# STEP 2 – DATA CLEANING
# ─────────────────────────────────────────────
print("─" * 55)
print("STEP 2 – Cleaning data…")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values: fill numeric columns with their median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Filled missing values in '{col}' with median")

# Fill missing text columns with mode
for col in df.select_dtypes(include=["object"]).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  Filled missing values in '{col}' with mode")

print("[OK] Data cleaning done!\n")

# ─────────────────────────────────────────────
# STEP 3 – DATE FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("─" * 55)
print("STEP 3 – Engineering date features…")

# Find the date column (common names)
date_col = None
for c in df.columns:
    if c.lower() in ["date", "week", "weekdate", "week_date"]:
        date_col = c
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
    df["Year"]        = df[date_col].dt.year
    df["Month"]       = df[date_col].dt.month
    df["Week"]        = df[date_col].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]   = df[date_col].dt.dayofweek
    df["Quarter"]     = df[date_col].dt.quarter
    df["IsWeekend"]   = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    df.drop(columns=[date_col], inplace=True)
    print(f"  Extracted Year, Month, Week, DayOfWeek, Quarter, IsWeekend from '{date_col}'")
else:
    print("  [SKIP] No date column found – using existing numeric columns")

print("[OK] Feature engineering done!\n")

# ─────────────────────────────────────────────
# STEP 4 – ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
print("─" * 55)
print("STEP 4 – Encoding categorical columns…")

le = LabelEncoder()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"  Encoded column: {col}")

print("[OK] Encoding done!\n")

# ─────────────────────────────────────────────
# STEP 5 – DEFINE FEATURES & TARGET
# ─────────────────────────────────────────────
# Target column: look for "Sales", "Weekly_Sales", etc.
target_col = None
for c in df.columns:
    if "sale" in c.lower():
        target_col = c
        break

if target_col is None:
    raise ValueError("Could not find a 'Sales' column in the dataset!")

print(f"[INFO] Target column → '{target_col}'\n")

X = df.drop(columns=[target_col])
y = df[target_col]

# ─────────────────────────────────────────────
# STEP 6 – TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}   Test size: {len(X_test)}\n")

# ─────────────────────────────────────────────
# STEP 7 – TRAIN 3 MODELS
# ─────────────────────────────────────────────
models = {
    "Linear Regression" : LinearRegression(),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost"           : XGBRegressor(n_estimators=100, learning_rate=0.1,
                                       random_state=42, verbosity=0),
}

results = {}

print("─" * 55)
print("STEP 7 – Training models…\n")

for name, model in models.items():
    print(f"  Training {name}…", end=" ")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"Done!  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

# ─────────────────────────────────────────────
# STEP 8 – COMPARE & PICK BEST MODEL
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("STEP 8 – Model comparison:\n")
print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("-" * 52)
for name, res in results.items():
    print(f"{name:<22} {res['MAE']:>10.2f} {res['RMSE']:>10.2f} {res['R2']:>8.4f}")

# Best model = highest R² score
best_name  = max(results, key=lambda n: results[n]["R2"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best model: {best_name}  (R² = {results[best_name]['R2']:.4f})\n")

# ─────────────────────────────────────────────
# STEP 9 – SAVE THE BEST MODEL
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "sales_model.pkl")
joblib.dump({"model": best_model, "features": list(X.columns)}, MODEL_PATH)
print(f"[OK] Model saved → {MODEL_PATH}\n")

# ─────────────────────────────────────────────
# STEP 10 – VISUALIZATIONS
# ─────────────────────────────────────────────
print("─" * 55)
print("STEP 10 – Saving visualizations…\n")

PLOT_DIR = os.path.join("static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── 10a. Sales distribution ──
plt.figure(figsize=(8, 4))
sns.histplot(y, bins=40, color="#4f8ef7", kde=True)
plt.title("Sales Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "sales_distribution.png"), dpi=120)
plt.close()
print("  Saved: sales_distribution.png")

# ── 10b. Actual vs Predicted (best model) ──
best_preds = best_model.predict(X_test)
plt.figure(figsize=(8, 4))
plt.plot(y_test.values[:100], label="Actual",    color="#4f8ef7", linewidth=2)
plt.plot(best_preds[:100],    label="Predicted", color="#f77f4f",
         linewidth=2, linestyle="--")
plt.title(f"Actual vs Predicted Sales ({best_name})", fontsize=14, fontweight="bold")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "actual_vs_predicted.png"), dpi=120)
plt.close()
print("  Saved: actual_vs_predicted.png")

# ── 10c. Model performance bar chart ──
names = list(results.keys())
r2s   = [results[n]["R2"]  for n in names]
maes  = [results[n]["MAE"] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ["#4f8ef7", "#f77f4f", "#4fc77f"]

axes[0].bar(names, r2s,  color=colors)
axes[0].set_title("R² Score (higher = better)", fontweight="bold")
axes[0].set_ylim(0, 1)
for i, v in enumerate(r2s):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

axes[1].bar(names, maes, color=colors)
axes[1].set_title("MAE (lower = better)", fontweight="bold")
for i, v in enumerate(maes):
    axes[1].text(i, v + max(maes)*0.01, f"{v:.1f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), dpi=120)
plt.close()
print("  Saved: model_comparison.png")

# ── 10d. Feature importance (if tree-based) ──
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True).tail(10)
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="#4f8ef7")
    plt.title("Top 10 Feature Importances", fontsize=14, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    print("  Saved: feature_importance.png")

print("\n✅  Training complete!  Run  python app.py  to start the web app.\n")
