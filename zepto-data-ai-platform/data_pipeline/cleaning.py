"""Step 2 - clean scraped text into typed columns and convert GBP -> INR.

Messy-row policy (never crash on bad input):
  * price_gbp  unparseable -> MEDIAN IMPUTATION. Price is a continuous numeric
    field; one bad string should not cost us the whole book, and the median is
    robust to the long tail of expensive titles.
  * rating     unparseable -> MEDIAN IMPUTATION, rounded to an int so the
    column stays a valid 1-5 star value.
  * in_stock   unparseable -> DROP ROW. A boolean has no meaningful median;
    guessing "in stock" would silently corrupt availability benchmarks.
  * title      missing      -> DROP ROW. A book with no identity can't be joined
    or reported on.
Every imputation/drop is counted and logged so it is visible in the run output.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

GBP_TO_INR = 105.50      # fixed, project-defined baseline rate (not a live/market rate)

RATING_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")

log = logging.getLogger(__name__)


def parse_price(text) -> float:
    """'£51.77' / 'Â£51.77' / '51.77' -> 51.77 ; anything else -> NaN."""
    if not isinstance(text, str):
        return np.nan
    match = _PRICE_RE.search(text.replace(",", ""))
    return float(match.group(1)) if match else np.nan


def parse_rating(text) -> float:
    """'Three' -> 3 ; unknown -> NaN (float so NaN is representable before imputation)."""
    if not isinstance(text, str):
        return np.nan
    return float(RATING_WORDS.get(text.strip().lower(), np.nan))


def parse_availability(text):
    """'In stock (22 available)' -> True, 'Out of stock' -> False, anything else -> None."""
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    if t.startswith("out of stock") or t.startswith("unavailable"):
        return False
    if t.startswith("in stock") or "available" in t:
        return True
    return None


def clean_books(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned DataFrame and a report of every cleaning action taken."""
    df = raw.copy()
    report: dict = {"rows_scraped": len(df)}

    df["title"] = df["title"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    df["price_gbp"] = df["price"].map(parse_price)
    df["rating"] = df["star_rating"].map(parse_rating)
    df["in_stock"] = df["availability"].map(parse_availability)

    # --- drops -------------------------------------------------------------
    bad_title = df["title"].isna() | (df["title"] == "")
    bad_stock = df["in_stock"].isna()
    report["dropped_missing_title"] = int(bad_title.sum())
    report["dropped_unparseable_availability"] = int((bad_stock & ~bad_title).sum())
    df = df[~bad_title & ~bad_stock].copy()

    # --- median imputation -------------------------------------------------
    report["imputed_price_gbp"] = int(df["price_gbp"].isna().sum())
    report["imputed_rating"] = int(df["rating"].isna().sum())
    if report["imputed_price_gbp"]:
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    if report["imputed_rating"]:
        df["rating"] = df["rating"].fillna(round(df["rating"].median()))

    # --- final types + currency conversion -----------------------------------
    df["price_gbp"] = df["price_gbp"].astype(float).round(2)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].reset_index(drop=True)
    report["rows_clean"] = len(df)
    report["categories"] = int(df["category"].nunique())

    assert df["rating"].between(1, 5).all(), "rating must be 1-5"
    assert np.allclose(df["price_inr"], (df["price_gbp"] * GBP_TO_INR).round(2))
    for k, v in report.items():
        log.info("cleaning: %-34s %s", k, v)
    return df, report
