"""Step 3 - normalised SQLite schema (two tables, PK/FK) and loading."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT    NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL CHECK (price_gbp >= 0),
    price_inr   REAL    NOT NULL CHECK (price_inr >= 0),
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);

CREATE INDEX idx_books_category ON books(category_id);
"""


def build_tables(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the flat cleaned frame into the two normalised tables (in memory)."""
    categories = (pd.DataFrame({"category_name": sorted(clean["category"].unique())})
                  .assign(category_id=lambda d: range(1, len(d) + 1))[["category_id", "category_name"]])
    books = (clean.merge(categories, left_on="category", right_on="category_name", how="left")
             .assign(book_id=lambda d: range(1, len(d) + 1),
                     in_stock=lambda d: d["in_stock"].astype(int))
             [["book_id", "title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]])
    return categories, books


def load_database(clean: pd.DataFrame, db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Re)create the database from scratch and insert both tables with sqlite3."""
    categories, books = build_tables(clean)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
                         categories.itertuples(index=False, name=None))
        conn.executemany(
            "INSERT INTO books (book_id, title, price_gbp, price_inr, rating, in_stock, category_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            books.itertuples(index=False, name=None))
        # FK integrity check: returns no rows when every books.category_id exists in categories
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    return categories, books
