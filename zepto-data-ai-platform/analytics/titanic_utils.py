"""Shared helpers for the /analytics module.

Both notebooks import from here so that the *row-level* cleaning decided in
01_eda.ipynb is applied identically in 02_modeling.ipynb (the dataset is
"cleaned once" in logic, even though each notebook runs in its own kernel).

Nothing in this file fits a statistic on the data (no means/medians are
learned here), so calling it before the train/test split cannot leak test
information into training.
"""
from __future__ import annotations

import pandas as pd

RAW_CSV = "titanic.csv"

# Colour roles used by every chart (validated categorical slots 1-3).
COLOR_SURVIVED = "#2a78d6"   # blue
COLOR_DIED = "#eb6834"       # orange
COLOR_THIRD = "#1baf7a"      # aqua
SURVIVAL_PALETTE = {0: COLOR_DIED, 1: COLOR_SURVIVED}

# Columns that are exact re-encodings of other columns (or of the target)
# and are therefore removed before any modelling.
REDUNDANT_COLUMNS = ["alive", "class", "embark_town", "who", "adult_male", "alone"]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of missing values for every column that has any."""
    pct = df.isna().mean().mul(100).round(2)
    pct = pct[pct > 0].sort_values(ascending=False)
    return pd.DataFrame({"missing_count": df[pct.index].isna().sum(), "missing_pct": pct})


def strategy_for(pct: float) -> str:
    """The project's threshold rule, expressed as code."""
    if pct < 5:
        return "drop rows (<5% missing)"
    if pct <= 30:
        return "impute (5%-30% missing)"
    return "too sparse to impute (>30%) -> encode 'Missing' as its own category"


def drop_sparse_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning shared by EDA and modelling.

    `embarked` / `embark_town` are missing for the same 2 passengers (0.22%),
    which falls under the <5% rule -> drop those rows. This is a deterministic
    row filter, not a fitted transform, so it is safe before the split.
    """
    return df.dropna(subset=["embarked"]).reset_index(drop=True)


def add_deck_category(df: pd.DataFrame) -> pd.DataFrame:
    """`deck` is ~77% missing -> keep it, but as an explicit 'Missing' level."""
    out = df.copy()
    out["deck"] = out["deck"].astype("object").fillna("Missing")
    return out
