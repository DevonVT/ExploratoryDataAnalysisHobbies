# hobbies_eda_fixed.py
# EDA on FactoMineR "hobbies" dataset
# Main fix: TV is NOT binary like the other hobbies, so we handle it separately.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------
DATA_PATH = r"C:\Homework_and_misc\CS5580\EDAassignment\data_MCA_Hobbies.csv"
sns.set_theme()

# ---------------------------------------------------------
# 2. LOAD (semicolon separated)
# ---------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find file at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=";")

print("=== SHAPE ===")
print(df.shape)   # should be (8403, 23)
print("=== COLUMNS ===")
print(df.columns.tolist())

# ---------------------------------------------------------
# 3. DEFINE COLUMN GROUPS
# first 18 are “activities”, but 17 are yes/no, 1 is TV (0–4 scale)
# ---------------------------------------------------------
all_activity_cols = df.columns[:18].tolist()
TV_COL = "TV"

# hobbies that really are yes/no
binary_hobby_cols = [c for c in all_activity_cols if c != TV_COL]

# ---------------------------------------------------------
# 4. CONVERT y/n → 1/0 FOR REAL BINARY HOBBIES
# ---------------------------------------------------------
df[binary_hobby_cols] = (
    df[binary_hobby_cols]
    .replace({"y": 1, "n": 0})
    .infer_objects(copy=False)
)

# TV we leave AS IS (it’s 0–4 already). Let’s make sure it’s numeric:
df[TV_COL] = pd.to_numeric(df[TV_COL], errors="coerce")

# ---------------------------------------------------------
# 5. CATEGORICALS
# ---------------------------------------------------------
df["Sex"] = df["Sex"].astype("category")
df["Marital status"] = df["Marital status"].astype("category")
df["Profession"] = df["Profession"].astype("category")

# ---------------------------------------------------------
# 6. AGE PARSING → midpoint
# dataset has ages like "(45,55]" or "45-55"
# ---------------------------------------------------------
def parse_age_to_mid(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    if "," in s:
        parts = s.split(",")
    else:
        parts = s.split("-")
    if len(parts) != 2:
        return None
    try:
        low = float(parts[0])
        high = float(parts[1])
        return (low + high) / 2
    except ValueError:
        return None

df["Age_mid"] = df["Age"].apply(parse_age_to_mid)

# ---------------------------------------------------------
# 7. NUMBER OF ACTIVITIES
# ---------------------------------------------------------
df["nb.activitees"] = pd.to_numeric(df["nb.activitees"], errors="coerce")

# ---------------------------------------------------------
# 8. QUICK CHECKS
# ---------------------------------------------------------
print("\n=== INFO ===")
print(df.info())
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# ---------------------------------------------------------
# 9. VISUALIZATIONS
# ---------------------------------------------------------

# 9.1 Participation rate per (binary) hobby — TV excluded here
plt.figure(figsize=(10, 4))
(
    df[binary_hobby_cols]
    .mean()
    .sort_values(ascending=False)
    .plot(kind="bar")
)
plt.title("Participation rate per hobby (TV excluded)")
plt.ylabel("Proportion of people (0–1)")
plt.tight_layout()
plt.show()

# 9.2 TV watching distribution (its own plot)
plt.figure(figsize=(5, 4))
df[TV_COL].value_counts().sort_index().plot(kind="bar")
plt.title("TV watching frequency (0 = never, 4 = very often)")
plt.xlabel("TV frequency code")
plt.ylabel("Number of people")
plt.tight_layout()
plt.show()

# 9.3 Correlation heatmap for hobbies only (TV excluded)
import numpy as np
plt.figure(figsize=(10, 8))
corr = df[binary_hobby_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation between hobbies (binary only)")
plt.tight_layout()
plt.show()

# 9.4 Number of hobbies by Sex
plt.figure(figsize=(6, 4))
sns.boxplot(x="Sex", y="nb.activitees", data=df)
plt.title("Number of hobbies by Sex")
plt.tight_layout()
plt.show()

# 9.5 Number of hobbies vs age
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Age_mid", y="nb.activitees", data=df, hue="Sex", alpha=0.6)
plt.title("Number of hobbies vs Age (midpoint)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. SUMMARY STATS FOR REPORT
# ---------------------------------------------------------
print("\n=== AVG # HOBBIES BY SEX ===")
print(df.groupby("Sex")["nb.activitees"].mean())

print("\n=== AVG # HOBBIES BY MARITAL STATUS ===")
print(df.groupby("Marital status")["nb.activitees"].mean())

print("\n=== CORR AGE vs # HOBBIES ===")
print(df["Age_mid"].corr(df["nb.activitees"]))

# Optional: average TV by sex (kind of fun to report)
print("\n=== AVG TV FREQUENCY BY SEX ===")
print(df.groupby("Sex")[TV_COL].mean())

# ---------------------------------------------------------
# 11. PRINT FIRST FEW LINES (for report appendix)
# ---------------------------------------------------------
print("\n=== FIRST 10 ROWS OF DATASET ===")
print(df.head(3).to_string(index=False))
