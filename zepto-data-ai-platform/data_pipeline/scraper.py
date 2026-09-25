"""Step 1 - scrape books.toscrape.com with requests + BeautifulSoup.

Scope chosen: every book in 4 whole categories (Mystery, Historical Fiction,
Romance, Sequential Art), following each category's "next" pagination links.
That is ~168 books, comfortably above the 60-book / 3-category minimum.

Only the category listing pages are fetched (not every product page): each
listing card already carries title, price, star rating and availability, so
this needs ~10 HTTP requests instead of ~170.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATEGORIES = ["Mystery", "Historical Fiction", "Romance", "Sequential Art"]
HEADERS = {"User-Agent": "zepto-capstone-data-pipeline/1.0 (educational scraping practice)"}
REQUEST_DELAY_S = 0.5       # be polite to a free practice site
TIMEOUT_S = 20

log = logging.getLogger(__name__)


def fetch_html(url: str, session: requests.Session, retries: int = 3) -> str:
    """GET a page, checking the status code explicitly and retrying transient failures."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=TIMEOUT_S)
            if resp.status_code == 200:
                resp.encoding = "utf-8"          # the site is UTF-8; avoids "Â£" mojibake in prices
                return resp.text
            log.warning("GET %s -> HTTP %s (attempt %d/%d)", url, resp.status_code, attempt, retries)
        except requests.RequestException as exc:
            log.warning("GET %s failed: %s (attempt %d/%d)", url, exc, attempt, retries)
        time.sleep(attempt)                      # simple linear back-off
    raise RuntimeError(f"Could not fetch {url} after {retries} attempts")


def find_category_urls(home_html: str, wanted: list[str]) -> dict[str, str]:
    """Map category name -> absolute URL of its first listing page, from the sidebar."""
    soup = BeautifulSoup(home_html, "html.parser")
    links = {a.get_text(strip=True): urljoin(BASE_URL, a["href"])
             for a in soup.select("div.side_categories ul li ul li a")}
    missing = [c for c in wanted if c not in links]
    if missing:
        raise ValueError(f"Categories not found on the site: {missing}")
    return {c: links[c] for c in wanted}


def parse_listing_page(html: str, category: str) -> tuple[list[dict], str | None]:
    """Extract raw (uncleaned) fields for every book card on one listing page.

    Returns the rows plus the relative href of the next page (or None).
    Values are kept exactly as listed; cleaning happens in cleaning.py.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select("article.product_pod"):
        title_tag = card.select_one("h3 a")
        price_tag = card.select_one("p.price_color")
        rating_tag = card.select_one("p.star-rating")
        avail_tag = card.select_one("p.availability")
        rating_classes = [c for c in (rating_tag.get("class", []) if rating_tag else []) if c != "star-rating"]
        rows.append({
            "title": title_tag.get("title", title_tag.get_text(strip=True)) if title_tag else None,
            "price": price_tag.get_text(strip=True) if price_tag else None,          # e.g. "£51.77"
            "star_rating": rating_classes[0] if rating_classes else None,           # e.g. "Three"
            "availability": avail_tag.get_text(" ", strip=True) if avail_tag else None,  # "In stock"
            "category": category,
        })
    next_link = soup.select_one("ul.pager li.next a")
    return rows, (next_link["href"] if next_link else None)


def scrape_category(name: str, first_url: str, session: requests.Session) -> list[dict]:
    rows, url, page = [], first_url, 1
    while url:
        page_rows, next_href = parse_listing_page(fetch_html(url, session), name)
        rows.extend(page_rows)
        log.info("  %-20s page %d -> %d books", name, page, len(page_rows))
        url = urljoin(url, next_href) if next_href else None      # next href is relative to current page
        page += 1
        time.sleep(REQUEST_DELAY_S)
    return rows


def scrape_books(categories: list[str] = CATEGORIES) -> list[dict]:
    with requests.Session() as session:
        category_urls = find_category_urls(fetch_html(BASE_URL, session), categories)
        all_rows: list[dict] = []
        for name, url in category_urls.items():
            all_rows.extend(scrape_category(name, url, session))
    log.info("Scraped %d books across %d categories", len(all_rows), len(categories))
    return all_rows
