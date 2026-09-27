from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
from typing import Any

import requests
import urllib3
from requests.exceptions import SSLError
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression


BASE_DIR = Path(__file__).resolve().parent

# Keep database outside the project folder so Live Server does not reload
# the frontend whenever market history is updated.
DB_PATH = Path(tempfile.gettempdir()) / "kabadiwala_market_history.db"

MSTC_URL = "https://etp.mstcindia.co.in/market"


app = FastAPI(
    title="Kabadiwala Connect Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,*/*;q=0.8"
    ),
}


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS market_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_date TEXT NOT NULL,
                material_key TEXT NOT NULL,
                category TEXT,
                subcategory TEXT,
                price_per_kg REAL NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                UNIQUE(observation_date, material_key)
            )
            """
        )
        con.commit()


init_db()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def slug(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "-",
        value.lower()
    ).strip("-")


def parse_number(value: Any) -> float | None:
    if value is None:
        return None

    raw = clean(value).replace(",", "")

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        raw
    )

    return float(match.group()) if match else None


def price_to_kg(
    price_text: Any,
    unit_text: Any = ""
) -> tuple[float | None, str]:

    raw = clean(price_text)

    price = parse_number(raw)

    if price is None:
        return None, ""

    unit_match = re.search(
        r"/\s*(kg|kilogram|ton|tonne|mt|metric\s*ton)\b",
        raw,
        re.I
    )

    if unit_match:
        unit = (
            unit_match.group(1)
            .lower()
            .replace(" ", "")
        )
    else:
        unit = clean(unit_text).lower()

    if unit in {
        "ton",
        "tonne",
        "mt",
        "metricton"
    }:
        return price / 1000.0, "Ton"

    if unit in {
        "kg",
        "kilogram"
    }:
        return price, "Kg"

    return price, ""


# ---------------------------------------------------------
# MSTC HTML PARSER
# ---------------------------------------------------------

def parse_mstc_html(
    html: str
) -> list[dict[str, Any]]:

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    items: list[dict[str, Any]] = []

    for table in soup.find_all("table"):

        rows = table.find_all("tr")

        if not rows:
            continue

        headers = [
            clean(
                cell.get_text(
                    " ",
                    strip=True
                )
            ).lower()
            for cell in rows[0].find_all(
                ["th", "td"]
            )
        ]

        if not headers:
            continue

        def find_index(*names):
            for index, header in enumerate(headers):
                normalized = header.replace(
                    "_",
                    " "
                )

                if any(
                    name in normalized
                    for name in names
                ):
                    return index

            return None

        category_index = find_index(
            "category"
        )

        subcategory_index = find_index(
            "sub cat",
            "subcategory",
            "sub category"
        )

        price_index = find_index(
            "last traded price",
            "traded price",
            "price"
        )

        volume_index = find_index(
            "volume",
            "quantity"
        )

        unit_index = find_index(
            "unit"
        )

        if price_index is None:
            continue

        for row in rows[1:]:

            cells = [
                clean(
                    cell.get_text(
                        " ",
                        strip=True
                    )
                )
                for cell in row.find_all(
                    ["td", "th"]
                )
            ]

            if (
                not cells
                or price_index >= len(cells)
            ):
                continue

            category = (
                cells[category_index]
                if (
                    category_index is not None
                    and category_index < len(cells)
                )
                else ""
            )

            subcategory = (
                cells[subcategory_index]
                if (
                    subcategory_index is not None
                    and subcategory_index < len(cells)
                )
                else ""
            )

            price_text = cells[price_index]

            unit_text = (
                cells[unit_index]
                if (
                    unit_index is not None
                    and unit_index < len(cells)
                )
                else ""
            )

            price, price_unit = price_to_kg(
                price_text,
                unit_text
            )

            if price is None or price <= 0:
                continue

            if not category and not subcategory:
                continue

            volume = (
                cells[volume_index]
                if (
                    volume_index is not None
                    and volume_index < len(cells)
                )
                else ""
            )

            volume_num = parse_number(volume)

            volume_unit = ""

            if volume_num is not None:
                volume_unit = clean(
                    re.sub(
                        r"[-+]?\d[\d,]*(?:\.\d+)?",
                        "",
                        volume
                    )
                ).strip()

            material_name = (
                f"{category}-{subcategory}"
            )

            items.append(
                {
                    "key": slug(material_name),
                    "category": category,
                    "subcategory": subcategory,
                    "price_per_kg": round(
                        price,
                        6
                    ),
                    "price_unit": (
                        price_unit
                        or "Unknown"
                    ),
                    "volume": volume_num,
                    "volume_unit": volume_unit,
                }
            )

    # Remove duplicate material entries.
    seen = set()
    output = []

    for item in items:

        key = (
            item["key"]
            or slug(
                item["subcategory"]
                or item["category"]
            )
        )

        if key in seen:
            continue

        seen.add(key)

        item["key"] = key

        output.append(item)

    return output


# ---------------------------------------------------------
# PLAYWRIGHT MSTC FETCH
# ---------------------------------------------------------

def fetch_mstc_rendered() -> list[dict[str, Any]]:

    if sync_playwright is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Playwright Python package is not installed."
            )
        )

    browser = None

    try:

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--ignore-certificate-errors",
                ],
            )

            page = browser.new_page(
                user_agent=HEADERS["User-Agent"],
                viewport={
                    "width": 1440,
                    "height": 1000,
                },
            )

            # IMPORTANT:
            # Do NOT use networkidle here.
            # MSTC keeps background requests active,
            # which can cause networkidle to wait forever.
            page.goto(
                MSTC_URL,
                wait_until="domcontentloaded",
                timeout=60000,
            )

            # Give the client-side application time to render.
            page.wait_for_timeout(8000)

            # First try normal table.
            try:
                page.wait_for_selector(
                    "table",
                    timeout=15000
                )
            except Exception:
                pass

            html = page.content()

            items = parse_mstc_html(html)

            if items:
                return items

            # Some MSTC responses can render the table later.
            page.wait_for_timeout(7000)

            html = page.content()

            items = parse_mstc_html(html)

            if items:
                return items

            return []

    except Exception as exc:

        print(
            "Playwright MSTC fetch error:",
            repr(exc)
        )

        return []

    finally:

        if browser is not None:

            try:
                browser.close()
            except Exception:
                pass


# ---------------------------------------------------------
# MSTC FETCH
# ---------------------------------------------------------

def fetch_mstc():

    # First try normal HTTP request.
    try:

        response = requests.get(
            MSTC_URL,
            headers=HEADERS,
            timeout=30,
            verify=True,
        )

        response.raise_for_status()

        static_items = parse_mstc_html(
            response.text
        )

        if static_items:

            return (
                static_items,
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

    except SSLError:

        urllib3.disable_warnings(
            urllib3.exceptions.InsecureRequestWarning
        )

        try:

            response = requests.get(
                MSTC_URL,
                headers=HEADERS,
                timeout=30,
                verify=False,
            )

            response.raise_for_status()

            static_items = parse_mstc_html(
                response.text
            )

            if static_items:

                return (
                    static_items,
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

        except requests.RequestException as exc:

            print(
                "MSTC TLS fallback error:",
                repr(exc)
            )

    except requests.RequestException as exc:

        print(
            "MSTC normal request error:",
            repr(exc)
        )

    # Static HTML did not contain the market table.
    # Use Chromium because MSTC is client-side rendered.
    items = fetch_mstc_rendered()

    if not items:

        raise HTTPException(
            status_code=502,
            detail=(
                "MSTC page was reached, but the market "
                "dashboard could not be parsed even after "
                "Chromium rendering."
            ),
        )

    return (
        items,
        datetime.now(
            timezone.utc
        ).isoformat()
    )


# ---------------------------------------------------------
# SAVE DAILY MARKET SNAPSHOT
# ---------------------------------------------------------

def save_snapshot(
    items: list[dict[str, Any]],
    fetched_at: str
):

    observation_date = (
        datetime.fromisoformat(
            fetched_at.replace(
                "Z",
                "+00:00"
            )
        )
        .date()
        .isoformat()
    )

    with db() as con:

        for item in items:

            con.execute(
                """
                INSERT INTO market_history
                (
                    observation_date,
                    material_key,
                    category,
                    subcategory,
                    price_per_kg,
                    source,
                    fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(
                    observation_date,
                    material_key
                )
                DO UPDATE SET
                    category=excluded.category,
                    subcategory=excluded.subcategory,
                    price_per_kg=excluded.price_per_kg,
                    source=excluded.source,
                    fetched_at=excluded.fetched_at
                """,
                (
                    observation_date,
                    item["key"],
                    item.get(
                        "category",
                        ""
                    ),
                    item.get(
                        "subcategory",
                        ""
                    ),
                    item["price_per_kg"],
                    MSTC_URL,
                    fetched_at,
                ),
            )

        con.commit()


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "ok": True,
        "service": "Kabadiwala Connect Backend",
        "mstc_source": MSTC_URL,
    }


# ---------------------------------------------------------
# LIVE MARKET PRICES
# ---------------------------------------------------------

@app.get("/api/market/prices")
def market_prices():

    items, fetched_at = fetch_mstc()

    save_snapshot(
        items,
        fetched_at
    )

    return {
        "source": MSTC_URL,
        "fetched_at": fetched_at,
        "items": items,
    }


# ---------------------------------------------------------
# PRICE FORECAST
# ---------------------------------------------------------

@app.get("/api/forecast")
def forecast(
    material: str = Query(
        ...,
        min_length=1
    )
):

    with db() as con:

        rows = con.execute(
            """
            SELECT
                observation_date,
                price_per_kg
            FROM market_history
            WHERE material_key = ?
            ORDER BY observation_date ASC
            """,
            (material,),
        ).fetchall()

    history = [
        {
            "date": row["observation_date"],
            "price_per_kg": row["price_per_kg"],
        }
        for row in rows
    ]

    # We intentionally do not generate fake forecasts.
    if len(history) < 3:

        return {
            "ready": False,
            "material": material,
            "history_points": len(history),
            "history": history,
            "source": MSTC_URL,
            "message": (
                "At least 3 different daily real "
                "observations are required before "
                "generating a forecast."
            ),
        }

    x = [
        [index]
        for index in range(
            len(history)
        )
    ]

    y = [
        float(point["price_per_kg"])
        for point in history
    ]

    model = LinearRegression().fit(
        x,
        y
    )

    future_x = [
        [len(history) + index]
        for index in range(1, 8)
    ]

    predictions = [
        max(
            0.0,
            float(value)
        )
        for value in model.predict(
            future_x
        )
    ]

    last_price = y[-1]

    forecast_rows = []

    for index, predicted in enumerate(
        predictions,
        start=1
    ):

        percent_change = (
            (
                predicted - last_price
            )
            / last_price
            * 100
            if last_price
            else 0.0
        )

        if percent_change > 1:
            trend = "up"
            label = "Rising"

        elif percent_change < -1:
            trend = "down"
            label = "Falling"

        else:
            trend = "flat"
            label = "Stable"

        forecast_date = (
            datetime.fromisoformat(
                history[-1]["date"]
            ).date()
            + timedelta(days=index)
        )

        forecast_rows.append(
            {
                "date": forecast_date.isoformat(),
                "predicted_price_per_kg": round(
                    predicted,
                    2
                ),
                "percent_change": round(
                    percent_change,
                    2
                ),
                "trend": trend,
                "trend_label": label,
            }
        )

    final_percent = (
        forecast_rows[-1]["percent_change"]
    )

    if final_percent > 1:

        advisory = (
            "Trend real observations ke "
            "basis par upar hai; local buyer "
            "rate aur material quality compare "
            "karke decision lein."
        )

    elif final_percent < -1:

        advisory = (
            "Trend real observations ke "
            "basis par neeche hai; better "
            "buyer quote compare karein aur "
            "quality/quantity factor dekhein."
        )

    else:

        advisory = (
            "Trend approximately stable hai; "
            "local buyer quotes aur material "
            "quality compare karke decision lein."
        )

    return {
        "ready": True,
        "material": material,
        "history_points": len(history),
        "history": history,
        "forecast": forecast_rows,
        "advisory": advisory,
        "source": MSTC_URL,
        "model": "LinearRegression",
    }
