# /data_pipeline — Catalogue scrape → clean → convert → SQLite → query

Part of the **Zepto Data & AI Platform** capstone. This module benchmarks catalogue-style price and
availability data before it reaches a dashboard. It scrapes the public practice site
[books.toscrape.com](https://books.toscrape.com/), cleans and types the fields, converts GBP to INR at the
project's fixed baseline rate, loads a normalised SQLite database, and queries it with both SQL and pandas.

## Files

| File | Responsibility |
|---|---|
| `scraper.py` | `requests` + `BeautifulSoup`: finds category URLs in the sidebar and follows each category's `next` pagination. Status codes are checked explicitly, with retries and a polite 0.5 s delay |
| `cleaning.py` | Text → typed columns (`price_gbp`, `rating`, `in_stock`), messy-row policy, `price_inr` conversion |
| `database.py` | Two-table schema (`categories` ← `books`, PK/FK), loads the data with `sqlite3`, runs an FK integrity check |
| `queries.py` | 7 SQL query strings, `pd.read_sql` examples, and the `pd.merge` re-implementation of the JOIN |
| `run_pipeline.py` | Orchestrates everything end to end and writes all outputs |
| `tests/test_pipeline.py` | Offline tests (synthetic HTML with the site's markup) for parsing, messy rows, and a full end-to-end run |
| `data/books_raw.csv`, `data/books_clean.csv` | Raw scraped text and cleaned typed data *(generated)* |
| `zepto_books.db` | The SQLite database *(generated; `run_pipeline.py` recreates it from scratch)* |
| `outputs/query_results.md` | Every query string with its printed output *(generated)* |

## How to run

```bash
cd data_pipeline
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py        # about 15 s: ~10 HTTP requests + processing
pytest -q                     # optional: offline tests, no network needed
```

`run_pipeline.py` needs no manual copy-pasting. It rebuilds `data/`, `zepto_books.db` and `outputs/query_results.md`
from scratch, prints every query and its result to the console, and rewrites the **Results** section at the bottom of this README.
If fewer than 60 books or 3 categories come back, it exits with an error.

## Scope

All books in **4 whole categories**: Mystery, Historical Fiction, Romance, Sequential Art. Sequential Art spans 4 listing
pages, and pagination is followed automatically. That gives roughly 168 books, well above the 60-book / 3-category minimum.
Only listing pages are fetched. Each listing card already shows title, price, star rating and availability, so this
takes about 10 requests instead of about 170 product-page requests. Fields captured per book: `title`, `price` (GBP, as listed),
`star_rating` (text, e.g. `"Three"`), `availability` (text), and `category`.

## Currency conversion

**Fixed project baseline rate: 1 GBP = 105.50 INR.** `price_inr = round(price_gbp × 105.50, 2)`.
This is an artificial, project-defined constant (`GBP_TO_INR` in `cleaning.py`), not a live or historical market
rate. It needs no API call and no network access. The optional live-rate stretch goal was not attempted, so the graded
fixed-rate path is the only conversion in the code.

## Cleaning and parsing decisions

| Field | Parsing | If it fails to parse | Why |
|---|---|---|---|
| `price_gbp` | Regex pulls the first number out of the text, so `"£51.77"` and the mojibake form `"Â£51.77"` both give `51.77`. The response is also forced to UTF-8 | **Median imputation** | Price is continuous. One bad string shouldn't cost a whole book, and the median is robust to expensive outliers |
| `rating` | `One…Five` → `1…5` (case-insensitive) | **Median imputation**, rounded to an int | Keeps the column a valid 1–5 integer |
| `in_stock` | Starts with "In stock" → `True`; "Out of stock"/"Unavailable" → `False` | **Drop the row** | A boolean has no meaningful median. Guessing "in stock" would silently corrupt an availability benchmark |
| `title` | Taken from the full `title` attribute of the link (the visible text is truncated with "…") | **Drop the row** | A book with no identity can't be reported on |

Every drop and imputation is counted, logged, and written to the run summary, so nothing is hidden. The pipeline never
crashes on a messy row. On the live site all fields parse cleanly, so the counters normally show 0. The logic is still
exercised by `tests/test_pipeline.py::test_messy_rows_are_imputed_or_dropped`, which uses deliberately broken rows.
Note that books.toscrape.com lists every book as "In stock", so `in_stock` is expected to be all `True`. Q7 still includes
the out-of-stock condition so it behaves correctly on real stock-outs.

## Schema (normalised, two tables, PK/FK)

```sql
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
```
The category name is stored once in `categories` rather than repeated on every book row (3NF). `PRAGMA foreign_keys = ON`
and a `PRAGMA foreign_key_check` after loading confirm referential integrity. CHECK constraints reject impossible values
at insert time. `in_stock` is stored as `0/1` because SQLite has no boolean type.

## Queries (all in `queries.py`)

| # | Purpose | Required clauses covered |
|---|---|---|
| Q1 | Top 10 most expensive in-stock books rated 4+ | SELECT, WHERE, ORDER BY, LIMIT |
| Q2 | Distinct star ratings present | DISTINCT |
| Q3 | Books priced GBP 20–30 with 1 or 2 stars | BETWEEN, IN |
| Q4 | Per-category count, average GBP/INR price, average rating | JOIN, GROUP BY |
| Q5 | All 5-star books with category name | JOIN (reproduced with `pd.merge`) |
| Q6 | Top 3 highest-rated books per category | JOIN + `ROW_NUMBER()` window |
| Q7 | Mystery/Romance books that are out of stock or under GBP 15 | JOIN, IN, LIMIT |

`pd.read_sql` reads Q4 and Q5 into DataFrames. Q5 is then rebuilt with **pandas only** (`pd.merge(books, categories,
on="category_id")`, then a filter for `rating == 5` and the same sort). `pd.testing.assert_frame_equal` proves the two
results are identical, and they are printed side by side.

## Results

The section below is written automatically by `python run_pipeline.py`: the run summary, each query string with its
output, the `pd.read_sql` DataFrames, and the side-by-side JOIN vs `pd.merge` comparison.

<!-- RESULTS:START -->
_Not generated yet. Run `python run_pipeline.py` to scrape the live site and fill this section._
<!-- RESULTS:END -->
