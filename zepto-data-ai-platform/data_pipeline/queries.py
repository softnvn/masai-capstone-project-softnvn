"""Step 4 - SQL queries, pd.read_sql, and the pd.merge equivalence check."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# Each query is saved as a string (and written to outputs/query_results.md with its output).
# The "clauses" list documents which required SQL features each one demonstrates.
QUERIES = [
    {
        "name": "Q1 - Top 10 most expensive in-stock books rated 4+ stars",
        "clauses": ["SELECT", "WHERE", "ORDER BY", "LIMIT"],
        "sql": """
SELECT title, price_gbp, price_inr, rating
FROM books
WHERE in_stock = 1 AND rating >= 4
ORDER BY price_gbp DESC, title
LIMIT 10;""",
    },
    {
        "name": "Q2 - Distinct star ratings present in the catalogue",
        "clauses": ["DISTINCT", "ORDER BY"],
        "sql": """
SELECT DISTINCT rating
FROM books
ORDER BY rating;""",
    },
    {
        "name": "Q3 - Mid-priced books (GBP 20-30) with low ratings (1 or 2 stars)",
        "clauses": ["WHERE", "BETWEEN", "IN", "ORDER BY"],
        "sql": """
SELECT title, price_gbp, price_inr, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 30
  AND rating IN (1, 2)
ORDER BY price_gbp;""",
    },
    {
        "name": "Q4 - Category benchmark: count, average price and average rating (JOIN + GROUP BY)",
        "clauses": ["JOIN", "GROUP BY", "ORDER BY"],
        "sql": """
SELECT c.category_name,
       COUNT(*)                     AS n_books,
       ROUND(AVG(b.price_gbp), 2)   AS avg_price_gbp,
       ROUND(AVG(b.price_inr), 2)   AS avg_price_inr,
       ROUND(AVG(b.rating), 2)      AS avg_rating,
       SUM(b.in_stock)              AS n_in_stock
FROM books AS b
JOIN categories AS c ON c.category_id = b.category_id
GROUP BY c.category_name
ORDER BY avg_price_gbp DESC;""",
    },
    {
        "name": "Q5 - All 5-star books with their category, highest price first (JOIN, reproduced with pd.merge)",
        "clauses": ["JOIN", "WHERE", "ORDER BY"],
        "sql": """
SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating
FROM books AS b
JOIN categories AS c ON c.category_id = b.category_id
WHERE b.rating = 5
ORDER BY c.category_name, b.price_gbp DESC, b.book_id;""",
    },
    {
        "name": "Q6 - Top 3 highest-rated books per category (JOIN + window function)",
        "clauses": ["JOIN", "WHERE", "ORDER BY"],
        "sql": """
SELECT category_name, title, rating, price_gbp
FROM (
    SELECT c.category_name, b.title, b.rating, b.price_gbp,
           ROW_NUMBER() OVER (PARTITION BY c.category_id
                              ORDER BY b.rating DESC, b.price_gbp DESC, b.book_id) AS rn
    FROM books AS b
    JOIN categories AS c ON c.category_id = b.category_id
)
WHERE rn <= 3
ORDER BY category_name, rn;""",
    },
    {
        "name": "Q7 - Books in the Mystery and Romance categories that are out of stock or cost under GBP 15",
        "clauses": ["JOIN", "WHERE", "IN", "ORDER BY", "LIMIT"],
        "sql": """
SELECT c.category_name, b.title, b.price_gbp, b.in_stock
FROM books AS b
JOIN categories AS c ON c.category_id = b.category_id
WHERE c.category_name IN ('Mystery', 'Romance')
  AND (b.in_stock = 0 OR b.price_gbp < 15)
ORDER BY b.price_gbp
LIMIT 10;""",
    },
]

JOIN_QUERY = QUERIES[4]       # Q5 - reproduced with pd.merge below


def run_queries(db_path: Path) -> list[tuple[dict, pd.DataFrame]]:
    """Execute every query with sqlite3 (cursor API) and return (query, result) pairs."""
    results = []
    with sqlite3.connect(db_path) as conn:
        for q in QUERIES:
            cur = conn.execute(q["sql"])
            cols = [d[0] for d in cur.description]
            results.append((q, pd.DataFrame(cur.fetchall(), columns=cols)))
    return results


def read_sql_examples(db_path: Path) -> dict[str, pd.DataFrame]:
    """Read two query results straight into DataFrames with pd.read_sql."""
    with sqlite3.connect(db_path) as conn:
        return {q["name"]: pd.read_sql(q["sql"], conn) for q in (QUERIES[3], JOIN_QUERY)}


def merge_equivalent(categories: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Reproduce JOIN_QUERY (Q5) with pandas only - no SQL involved."""
    merged = pd.merge(books, categories, on="category_id", how="inner")
    out = merged.loc[merged["rating"] == 5,
                     ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating"]]
    return (out.sort_values(["category_name", "price_gbp", "book_id"], ascending=[True, False, True])
            .reset_index(drop=True))


def compare_join(sql_df: pd.DataFrame, merge_df: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """Check both approaches give identical output and build a side-by-side view."""
    a = sql_df.reset_index(drop=True)
    b = merge_df.reset_index(drop=True).astype(a.dtypes.to_dict())
    pd.testing.assert_frame_equal(a, b, check_dtype=False)          # raises if anything differs
    side = pd.concat({"pd.read_sql (SQL JOIN)": a[["title", "category_name", "price_gbp"]],
                      "pd.merge (pandas only)": b[["title", "category_name", "price_gbp"]]}, axis=1)
    return a.equals(b), side
