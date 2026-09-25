# Zepto Data & AI Platform — Capstone Project

**Certificate Program in Artificial Intelligence and Machine Learning, Capstone Project**

One repository, three connected capabilities, one story. Raw catalogue data is scraped and turned into a clean relational
store for Zepto's analysts (**/data_pipeline**). A customer-style dataset is profiled and modelled end to end
(**/analytics**). Finally, a grounded GenAI support assistant answers questions about Zepto's own policies
(**/support_assistant**).

| Module | Marks | What it delivers | Module README |
|---|---|---|---|
| [`/data_pipeline`](data_pipeline/) | 25 | `requests` + `BeautifulSoup` scrape of books.toscrape.com → typed cleaning → GBP→INR at the fixed rate → normalised SQLite (PK/FK) → 7 SQL queries + `pd.read_sql` + `pd.merge` equivalence | [data_pipeline/README.md](data_pipeline/README.md) |
| [`/analytics`](analytics/) | 50 | Titanic: profiling, threshold-based cleaning, univariate/bivariate/multivariate EDA story, stratified split, leak-free `Pipeline`, 3 classifiers, imbalance comparison, `GridSearchCV` + OOB, regression, saved pipeline | [analytics/README.md](analytics/README.md) |
| [`/support_assistant`](support_assistant/) | 25 | 8 policy docs → MiniLM embeddings → ChromaDB → LangGraph intent router → Pydantic JSON → FastAPI `POST /ask` → Dockerfile; deterministic offline `MOCK_LLM` mode | [support_assistant/README.md](support_assistant/README.md) |

```
.
├── README.md                     ← you are here: setup, how to run, design decisions
├── .gitignore
├── data_pipeline/
│   ├── scraper.py  cleaning.py  database.py  queries.py  run_pipeline.py
│   ├── tests/test_pipeline.py
│   ├── data/ (books_raw.csv, books_clean.csv)   zepto_books.db   outputs/query_results.md   ← generated
│   ├── requirements.txt
│   └── README.md
├── analytics/
│   ├── 01_eda.ipynb  02_modeling.ipynb  titanic_utils.py
│   ├── titanic.csv  titanic_best_pipeline.joblib  model_comparison.csv  figures/*.png
│   ├── requirements.txt
│   └── README.md
└── support_assistant/
    ├── docs/doc_01.txt … doc_08.txt
    ├── rag.py  prompts.py  llm.py  graph.py  schemas.py  main.py  demo_calls.py
    ├── tests/test_assistant.py
    ├── Dockerfile  .dockerignore  requirements.txt
    └── README.md
```

## Setup

**Requirements: one `requirements.txt` per module** (not a single consolidated file). The modules have very different
dependency weights. The support assistant pulls in PyTorch through sentence-transformers, and its Docker image should
only contain what it needs. Keeping the files separate lets each module be installed and run on its own.

Python 3.10–3.12 is recommended (developed on 3.11). One virtual environment per module is cleanest:

```bash
git clone <this-repo-url> zepto-data-ai-platform && cd zepto-data-ai-platform

python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt            # large: includes torch
```
(A single shared environment also works; the three files have no conflicting pins.)

No paid service, API key or account is needed anywhere. Network access is needed only for scraping books.toscrape.com, for
the first `sns.load_dataset('titanic')` (there is a committed CSV fallback), and for the one-time download of the
`all-MiniLM-L6-v2` model (~90 MB, from Hugging Face).

## How to run each module end to end

### 1. Data pipeline
```bash
cd data_pipeline
python run_pipeline.py      # scrape → clean → convert → SQLite → queries; writes data/, zepto_books.db, outputs/
pytest -q                   # offline tests
```
The currency conversion uses the **fixed project baseline rate 1 GBP = 105.50 INR**. No API call is made.

### 2. Analytics
```bash
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb        # the one and only raw load; saves titanic.csv
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb   # reads titanic.csv; trains, tunes, saves pipeline
```
Or open them in Jupyter and run all cells in order. Both notebooks are committed **already executed**, with outputs.

### 3. Support assistant
```bash
cd support_assistant
uvicorn main:app --host 0.0.0.0 --port 7860           # MOCK_LLM unset = graded offline mock mode
python demo_calls.py --url http://127.0.0.1:7860      # in a second terminal; records JSON responses in its README
pytest -q

docker build -t zepto-support-assistant .             # container baseline
docker run --rm -p 7860:7860 zepto-support-assistant
```

## Design decisions (summary)

### /data_pipeline
- **Scope:** 4 full categories (Mystery, Historical Fiction, Romance, Sequential Art; about 168 books), with pagination
  followed automatically. Only listing pages are scraped, because each card already has every required field
  (about 10 requests instead of about 170).
- **Robust fetching:** status codes are checked explicitly, with retries, back-off, a polite delay and forced UTF-8
  decoding (so prices never come through as `Â£`).
- **Messy rows:** numeric fields (`price_gbp`, `rating`) are **median-imputed**. Rows with unparseable `in_stock` or a
  missing title are **dropped**, because a boolean has no median and guessing availability would corrupt the benchmark.
  Every action is counted and reported.
- **Schema:** `categories(category_id PK, category_name UNIQUE)` ← `books(..., category_id FK)`, with CHECK constraints
  and an FK integrity check after loading. The database is rebuilt from scratch on every run.
- **Queries:** 7 queries cover SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN, JOIN, GROUP BY and a window
  function. The JOIN query is reproduced with `pd.merge`, and `assert_frame_equal` proves the two results match.
- **Reproducibility:** one command regenerates every output and rewrites the module README's results section. Offline
  tests use synthetic HTML that mimics the site's markup.

### /analytics
- **Load once:** `sns.load_dataset('titanic')` is called once, in the first code cell of `01_eda.ipynb`, and
  immediately saved to `titanic.csv`. `02_modeling.ipynb` reads that CSV. Shared row-level rules live in `titanic_utils.py`.
- **Missing values (threshold rule):** `embarked`/`embark_town` 0.22% → drop 2 rows. `age` 19.87% → impute with the
  median of the passenger's (sex, pclass) group. `deck` 77.22% → an explicit `"Missing"` category, because the
  missingness itself carries signal (66.7% vs 29.9% survival).
- **Leak-free modelling:** stratified 80/20 split first. After that, every imputer, encoder and scaler sits inside a
  `ColumnTransformer` wrapped in a `Pipeline`, and SMOTE sits inside an `imblearn` pipeline, so fitting on test data is
  structurally impossible.
- **Results:** the tuned Random Forest (`max_depth=8, max_features=None, n_estimators=100`; OOB 0.827) is deployed:
  test accuracy 0.826, F1 0.760, precision 0.803. Logistic Regression is the documented runner-up, with the best
  AUC (0.861). The saved artifact is the whole fitted pipeline, and a reload is verified on raw rows, including a
  missing age.
- **Honest reporting:** outlier counts are given before and after imputation. The fare regression's heteroscedasticity
  is stated plainly (R² 0.35), together with the fix that would address it.

### /support_assistant
- **Mock-first:** a single `MOCK_LLM` toggle (`llm.is_mock()`) is read at call time. In the default mode no LLM code
  path runs, and the tests fail if `requests.post` is ever called.
- **Chunking:** 1 sentence per chunk (26 chunks). Each sentence is one policy rule, so the top chunk's first 200
  characters, which the mock template returns, actually contain the answer.
- **Retrieval:** always real: local `all-MiniLM-L6-v2` embeddings with cosine distance in the ChromaDB collection
  `zepto_policies`, top-3.
- **Graph:** a `TypedDict` state and 3 nodes. The conditional edge depends only on `intent`. Only the generation step
  in each node branches on `MOCK_LLM`.
- **Schema guarantee:** a strict Pydantic `AskResponse` (`extra="forbid"`, confidence in [0, 1]). On the real-LLM path
  the output is JSON-extracted and validated, retried up to 2 more times with a corrective instruction, then returned as
  a marked `[ERROR]` response. Sources the model invents are filtered out.
- **Container:** CPU-only torch, the model baked in at build time (the container starts offline), a non-root uid 1000
  user (compatible with HF Spaces), and a health check.

## Git workflow

The repository history uses feature branches merged back into `main` with merge commits (`--no-ff`):
`feature/data-pipeline`, `feature/analytics` and `feature/support-assistant`, each committed to at least twice before
merging. You can see this with:
```bash
git log --graph --oneline --all
```
