"""Offline tests for the data pipeline (run: `pytest -q` inside /data_pipeline).

The HTML below is a small *synthetic* page that mirrors the markup structure of
books.toscrape.com (sidebar, product_pod cards, pager). It lets the parsing,
cleaning, database and query logic be tested without network access. It is
never used by run_pipeline.py and none of its rows end up in the real database.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cleaning  # noqa: E402
import queries  # noqa: E402
import run_pipeline  # noqa: E402
import scraper  # noqa: E402

WORDS = ["One", "Two", "Three", "Four", "Five"]
HOME = """<html><body><div class="side_categories"><ul><li><a href="#">Books</a><ul>
<li><a href="catalogue/category/books/mystery_3/index.html">Mystery</a></li>
<li><a href="catalogue/category/books/historical-fiction_4/index.html">Historical Fiction</a></li>
<li><a href="catalogue/category/books/romance_8/index.html">Romance</a></li>
<li><a href="catalogue/category/books/sequential-art_5/index.html">Sequential Art</a></li>
</ul></li></ul></div></body></html>"""


def card(title, price, rating, avail="In stock"):
    return (f'<article class="product_pod"><p class="star-rating {rating}"></p>'
            f'<h3><a href="x.html" title="{title}">{title[:10]}...</a></h3>'
            f'<div class="product_price"><p class="price_color">{price}</p>'
            f'<p class="instock availability"><i class="icon-ok"></i> {avail} </p></div></article>')


def listing(cat, page, n, has_next):
    cards = "".join(card(f"{cat} Book {page}-{i}", f"£{10 + (i * 7 + page * 3) % 45}.{i % 10}9",
                         WORDS[(i + page) % 5]) for i in range(n))
    pager = '<ul class="pager"><li class="next"><a href="page-2.html">next</a></li></ul>' if has_next else ""
    return f"<html><body><section>{cards}</section>{pager}</body></html>"


@pytest.fixture
def fake_site(monkeypatch):
    def fake_fetch(url, session, retries=3):
        if url == scraper.BASE_URL:
            return HOME
        cat = next(c for c in scraper.CATEGORIES if c.lower().replace(" ", "-") in url)
        page = 2 if url.endswith("page-2.html") else 1
        return listing(cat, page, 20 if page == 1 else 5, has_next=(page == 1))
    monkeypatch.setattr(scraper, "fetch_html", fake_fetch)
    monkeypatch.setattr(scraper, "REQUEST_DELAY_S", 0)


def test_parsers():
    assert cleaning.parse_price("£51.77") == 51.77
    assert cleaning.parse_price("Â£13.99") == 13.99
    assert pd.isna(cleaning.parse_price("free!"))
    assert cleaning.parse_rating("Three") == 3 and pd.isna(cleaning.parse_rating("Zero"))
    assert cleaning.parse_availability("In stock (22 available)") is True
    assert cleaning.parse_availability("Out of stock") is False
    assert cleaning.parse_availability("Unavailable") is False
    assert cleaning.parse_availability("???") is None


def test_messy_rows_are_imputed_or_dropped():
    raw = pd.DataFrame([
        {"title": "A", "price": "£10.00", "star_rating": "One", "availability": "In stock", "category": "X"},
        {"title": "B", "price": "£30.00", "star_rating": "Five", "availability": "In stock", "category": "X"},
        {"title": "C", "price": "N/A", "star_rating": "Three", "availability": "In stock", "category": "Y"},
        {"title": "D", "price": "£20.00", "star_rating": "???", "availability": "In stock", "category": "Y"},
        {"title": "E", "price": "£20.00", "star_rating": "Two", "availability": "maybe", "category": "Y"},
        {"title": None, "price": "£5.00", "star_rating": "Two", "availability": "In stock", "category": "Y"},
    ])
    clean, rep = cleaning.clean_books(raw)
    assert rep["dropped_missing_title"] == 1 and rep["dropped_unparseable_availability"] == 1
    assert rep["imputed_price_gbp"] == 1 and rep["imputed_rating"] == 1
    assert clean.loc[clean.title == "C", "price_gbp"].item() == 20.0          # median of 10, 30, 20
    assert clean.loc[clean.title == "D", "rating"].item() == 3                 # median of 1, 5, 3
    assert clean["price_inr"].tolist() == [round(p * 105.50, 2) for p in clean["price_gbp"]]
    assert str(clean["rating"].dtype).startswith("int") and clean["in_stock"].dtype == bool


def test_end_to_end_offline(fake_site, tmp_path, monkeypatch):
    for name, value in {"DATA_DIR": tmp_path / "data", "OUT_DIR": tmp_path / "out",
                        "DB_PATH": tmp_path / "t.db", "README": tmp_path / "README.md"}.items():
        monkeypatch.setattr(run_pipeline, name, value)
    (tmp_path / "README.md").write_text(f"intro\n{run_pipeline.START}\nSTALE_PLACEHOLDER\n{run_pipeline.END}\nfooter\n")

    assert run_pipeline.main([]) == 0

    with sqlite3.connect(tmp_path / "t.db") as conn:
        n_books, n_cats = conn.execute("SELECT COUNT(*), COUNT(DISTINCT category_id) FROM books").fetchone()
        fk = conn.execute("PRAGMA foreign_key_list(books)").fetchall()
    assert n_books == 100 and n_cats == 4          # 4 categories x (20 + 5) synthetic cards
    assert fk and fk[0][2] == "categories"
    readme = (tmp_path / "README.md").read_text()
    assert "STALE_PLACEHOLDER" not in readme and "footer" in readme and "identical: **True**" in readme
    covered = {c for q in queries.QUERIES for c in q["clauses"]}
    assert {"SELECT", "WHERE", "ORDER BY", "LIMIT", "DISTINCT", "JOIN"} <= covered
    assert {"IN", "BETWEEN"} & covered
