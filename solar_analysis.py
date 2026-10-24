"""
Voyage-123 Solar Radiation: Analysis and Prediction
===================================================
Tasks:
  1. Data visualisation
  2. Inferences (peak hour, correlations)
  3. Solar day analysis (daily report, peak day, top-5 days)
  4. Machine-learning prediction (Random Forest)

Usage:  python solar_analysis.py [path/to/SolarPrediction.csv]
Plots are saved to ./plots/
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend: save figures without needing a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# 0. LOAD + CLEAN DATA
# --------------------------------------------------------------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "SolarPrediction.csv"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()  # remove stray whitespace in headers
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# --- Type conversions -------------------------------------------------------
# 'Data' looks like "9/29/2016 12:00:00 AM" (the time part is a dummy), so we
# keep only the date portion; the real clock time lives in 'Time'.
df["Data"] = df["Data"].astype(str).str.split().str[0]
df["Datetime"] = pd.to_datetime(
    df["Data"] + " " + df["Time"].astype(str), format="%m/%d/%Y %H:%M:%S", errors="coerce"
)
df["Date"] = df["Datetime"].dt.normalize()

# Sunrise / sunset are clock times ("06:13:00"); attach them to each row's date
# so we can do datetime arithmetic with them.
for col in ["TimeSunRise", "TimeSunSet"]:
    df[col + "_dt"] = pd.to_datetime(
        df["Data"] + " " + df[col].astype(str), format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )

# Make sure numeric columns really are numeric (bad values become NaN)
num_cols = ["Radiation", "Temperature", "Pressure", "Humidity",
            "WindDirection(Degrees)", "Speed"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Missing values ---------------------------------------------------------
print("\nMissing values before cleaning:\n", df[num_cols + ["Datetime"]].isna().sum())
df = df.dropna(subset=["Datetime", "Radiation"])           # rows unusable without these
df[num_cols] = df[num_cols].fillna(df[num_cols].median())  # impute remaining gaps
df = df.drop_duplicates().sort_values("Datetime").reset_index(drop=True)

# Handy time features
df["Hour"] = df["Datetime"].dt.hour
df["Month"] = df["Datetime"].dt.month
df["DayOfYear"] = df["Datetime"].dt.dayofyear
df["MinuteOfDay"] = df["Datetime"].dt.hour * 60 + df["Datetime"].dt.minute
df["SunriseMin"] = df["TimeSunRise_dt"].dt.hour * 60 + df["TimeSunRise_dt"].dt.minute
df["SunsetMin"] = df["TimeSunSet_dt"].dt.hour * 60 + df["TimeSunSet_dt"].dt.minute
df["DaylightMin"] = df["SunsetMin"] - df["SunriseMin"]
df["MinSinceSunrise"] = df["MinuteOfDay"] - df["SunriseMin"]

# --------------------------------------------------------------------------
# 1. DATA VISUALISATION
# --------------------------------------------------------------------------
print("\n" + "=" * 70 + "\nTASK 1: DATA VISUALISATION\n" + "=" * 70)

# 1a. Radiation over the whole period (raw + daily mean/max)
daily = df.groupby("Date")["Radiation"].agg(["mean", "max"])
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["Datetime"], df["Radiation"], lw=0.3, alpha=0.5, label="Raw observations")
ax.plot(daily.index + pd.Timedelta(hours=12), daily["mean"], color="red", lw=1.5, label="Daily mean")
ax.set(title="Solar Radiation over Time", xlabel="Date", ylabel="Radiation (W/m²)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/01_radiation_over_time.png", dpi=150)
plt.close(fig)

# 1b. Radiation vs temperature and humidity (scatter), coloured by hour
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ["Temperature", "Humidity"]):
    sc = ax.scatter(df[col], df["Radiation"], c=df["Hour"], cmap="viridis", s=4, alpha=0.5)
    ax.set(title=f"Radiation vs {col}", xlabel=col, ylabel="Radiation (W/m²)")
fig.colorbar(sc, ax=axes, label="Hour of day")
fig.savefig(f"{PLOT_DIR}/02_radiation_vs_temp_humidity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 1c. Average diurnal cycle (radiation by hour)
hourly = df.groupby("Hour")["Radiation"].mean()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=ax)
ax.set(title="Average Radiation by Hour of Day", xlabel="Hour", ylabel="Mean radiation (W/m²)")
ax.set_xticks(range(0, 24))
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/03_hourly_profile.png", dpi=150)
plt.close(fig)

# 1d. Outlier detection: boxplot of radiation by hour
fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(data=df, x="Hour", y="Radiation", fliersize=2, ax=ax)
ax.set(title="Radiation Distribution by Hour (outliers shown as points)",
       xlabel="Hour", ylabel="Radiation (W/m²)")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/04_outliers_boxplot.png", dpi=150)
plt.close(fig)

# 1e. Bonus: hour x month heatmap of mean radiation
pivot = df.pivot_table(index="Hour", columns="Month", values="Radiation", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, cbar_kws={"label": "Mean radiation (W/m²)"}, ax=ax)
ax.set(title="Mean Radiation: Hour of Day vs Month")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/05_bonus_hour_month_heatmap.png", dpi=150)
plt.close(fig)

# 1f. Correlation heatmap (used in Task 2)
corr_cols = ["Radiation", "Temperature", "Pressure", "Humidity",
             "WindDirection(Degrees)", "Speed", "Hour"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/06_correlation_heatmap.png", dpi=150)
plt.close(fig)

print(f"Saved 6 plots to ./{PLOT_DIR}/")

# --------------------------------------------------------------------------
# 2. INFERENCES
# --------------------------------------------------------------------------
print("\n" + "=" * 70 + "\nTASK 2: INFERENCES\n" + "=" * 70)

peak_hour = hourly.idxmax()
print(f"Hour with highest average radiation: {peak_hour}:00 - {peak_hour}:59 "
      f"({hourly.max():.1f} W/m²)")

print("\nCorrelation matrix:\n", corr.round(3))

# Rank predictors by absolute correlation with Radiation (exclude itself)
rad_corr = corr["Radiation"].drop("Radiation")
print("\nCorrelation with Radiation (sorted by strength):")
print(rad_corr.reindex(rad_corr.abs().sort_values(ascending=False).index).round(3))
env_only = rad_corr.drop("Hour")
print(f"\nStrongest ENVIRONMENTAL predictor: {env_only.abs().idxmax()} "
      f"(r = {env_only[env_only.abs().idxmax()]:.3f})")

# Outliers via the IQR rule on night-time vs day-time radiation
q1, q3 = df["Radiation"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print(f"\nIQR upper fence: {upper:.1f} W/m²; "
      f"{(df['Radiation'] > upper).sum()} readings exceed it (mostly midday peaks).")
print(f"Humidity readings > 100%: {(df['Humidity'] > 100).sum()} "
      f"(physically odd; likely sensor over-reading)")

# --------------------------------------------------------------------------
# 3. SOLAR DAY ANALYSIS
# --------------------------------------------------------------------------
print("\n" + "=" * 70 + "\nTASK 3: SOLAR DAY ANALYSIS\n" + "=" * 70)

report = (df.groupby("Date")["Radiation"]
            .agg(Max_Radiation="max", Avg_Radiation="mean")
            .round(2))
print("Day-wise report (first 5 days):\n", report.head())

# Row with the single highest radiation value
peak_row = df.loc[df["Radiation"].idxmax()]
peak_date = peak_row["Date"]
print(f"\nDate with highest recorded radiation: {peak_date.date()}")
print(f"  Peak radiation : {peak_row['Radiation']:.2f} W/m²")
print(f"  Time of peak   : {peak_row['Datetime'].time()}")
print(f"  Temperature    : {peak_row['Temperature']:.0f}")
print(f"  Humidity       : {peak_row['Humidity']:.0f} %")
print(f"  Wind speed     : {peak_row['Speed']:.2f}")

# Time elapsed between sunrise and the peak
sunrise, sunset = peak_row["TimeSunRise_dt"], peak_row["TimeSunSet_dt"]
since_sunrise = peak_row["Datetime"] - sunrise
hrs, rem = divmod(int(since_sunrise.total_seconds()), 3600)
print(f"  Sunrise        : {sunrise.time()}")
print(f"  Sunset         : {sunset.time()}")
print(f"  Peak occurred {hrs} h {rem // 60} min after sunrise "
      f"({since_sunrise / (sunset - sunrise):.0%} of the way through daylight)")

# Top 5 days by average radiation
print("\nTop 5 days by average radiation:")
print(report.sort_values("Avg_Radiation", ascending=False).head(5))

# --------------------------------------------------------------------------
# 4. MACHINE LEARNING: RANDOM FOREST REGRESSION
# --------------------------------------------------------------------------
print("\n" + "=" * 70 + "\nTASK 4: RANDOM FOREST PREDICTION\n" + "=" * 70)

# Features: environmental readings plus temporal features. Sunrise/sunset are
# folded in via MinSinceSunrise and DaylightMin (position within the solar day).
features = ["Temperature", "Pressure", "Humidity", "WindDirection(Degrees)", "Speed",
            "Hour", "MinuteOfDay", "DayOfYear", "MinSinceSunrise", "DaylightMin"]
X, y = df[features], df["Radiation"]

# Random split. Note: neighbouring readings are 5 min apart, so a random split
# is optimistic compared with forecasting genuinely unseen days.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=2,
                              n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"RMSE: {rmse:.2f} W/m²")
print(f"R²  : {r2:.4f}")

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:\n", importances.round(4))

# Stricter check: hold out the last 20% of days chronologically
cut = int(len(df) * 0.8)
m2 = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)
m2.fit(X.iloc[:cut], y.iloc[:cut])
p2 = m2.predict(X.iloc[cut:])
print(f"\nChronological hold-out -> RMSE: {np.sqrt(mean_squared_error(y.iloc[cut:], p2)):.2f}, "
      f"R²: {r2_score(y.iloc[cut:], p2):.4f}")

# Plots: importances and predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=importances.values, y=importances.index, ax=axes[0], color="steelblue")
axes[0].set(title="Random Forest Feature Importances", xlabel="Importance")
axes[1].scatter(y_test, pred, s=4, alpha=0.4)
lim = [0, max(y_test.max(), pred.max())]
axes[1].plot(lim, lim, "r--", label="Perfect prediction")
axes[1].set(title=f"Predicted vs Actual (R²={r2:.3f})", xlabel="Actual", ylabel="Predicted")
axes[1].legend()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/07_model_results.png", dpi=150)
plt.close(fig)
print(f"\nSaved model plot to ./{PLOT_DIR}/07_model_results.png")
