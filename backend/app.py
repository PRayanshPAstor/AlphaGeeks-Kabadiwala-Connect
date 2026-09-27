from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
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
DB_PATH = Path(tempfile.gettempdir()) / "kabadiwala_market_history.db"
MSTC_URL = "https://etp.mstcindia.co.in/market"

app = FastAPI(title="Kabadiwala Connect Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


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


def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def parse_number(s: Any) -> float | None:
    if s is None:
        return None
    raw = clean(s).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    return float(m.group()) if m else None


def price_to_kg(price_text: Any, unit_text: Any = "") -> tuple[float | None, str]:
    """Convert MSTC's displayed price (e.g. ₹5400/Ton or ₹50/Kg) to ₹/kg.

    MSTC puts the price unit inside the price cell. We therefore parse that
    unit first instead of guessing it from the volume column.
    """
    raw = clean(price_text)
    price = parse_number(raw)
    if price is None:
        return None, ""

    m = re.search(r"/\s*(kg|kilogram|ton|tonne|mt|metric\s*ton)\b", raw, re.I)
    unit = (m.group(1).lower().replace(" ", "") if m else clean(unit_text).lower())

    if unit in {"ton", "tonne", "mt", "metricton"}:
        return price / 1000.0, "Ton"
    if unit in {"kg", "kilogram"}:
        return price, "Kg"
    return price, ""


def parse_mstc_html(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        headers = [clean(c.get_text(" ", strip=True)).lower() for c in rows[0].find_all(["th", "td"])]
        if not headers:
            continue

        def idx(*names):
            for i, h in enumerate(headers):
                normalized = h.replace("_", " ")
                if any(n in normalized for n in names):
                    return i
            return None

        cat_i = idx("category")
        sub_i = idx("sub cat", "subcategory", "sub category")
        price_i = idx("last traded price", "traded price", "price")
        volume_i = idx("volume", "quantity")
        unit_i = idx("unit")

        if price_i is None:
            continue

        for tr in rows[1:]:
            cells = [clean(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
            if not cells or price_i >= len(cells):
                continue
            category = cells[cat_i] if cat_i is not None and cat_i < len(cells) else ""
            subcategory = cells[sub_i] if sub_i is not None and sub_i < len(cells) else ""
            price_text = cells[price_i]
            # The MSTC price cell itself contains the authoritative unit, e.g.
            # "₹ 5400/Ton" or "₹ 50/Kg". Do not use the volume unit as the
            # price unit because a volume column can be a different quantity.
            unit_text = cells[unit_i] if unit_i is not None and unit_i < len(cells) else ""
            price, price_unit = price_to_kg(price_text, unit_text)
            if price is None or price <= 0:
                continue
            if not category and not subcategory:
                continue
            volume = cells[volume_i] if volume_i is not None and volume_i < len(cells) else ""
            volume_num = parse_number(volume)
            volume_unit = ""
            if volume_num is not None:
                volume_unit = clean(re.sub(r"[-+]?\d[\d,]*(?:\.\d+)?", "", volume)).strip()
            name = subcategory or category
            items.append({
                "key": slug(f"{category}-{subcategory}"),
                "category": category,
                "subcategory": subcategory,
                "price_per_kg": round(price, 6),
                "price_unit": price_unit or "Unknown",
                "volume": volume_num,
                "volume_unit": volume_unit,
            })

    # Remove duplicates while preserving order.
    seen = set()
    out = []
    for item in items:
        key = item["key"] or slug(item["subcategory"] or item["category"])
        if key in seen:
            continue
        seen.add(key)
        item["key"] = key
        out.append(item)
    return out


def fetch_mstc_rendered() -> list[dict[str, Any]]:
    """Render the MSTC page in Chromium because its market dashboard is JS-driven."""
    if sync_playwright is None:
        return []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--ignore-certificate-errors"],
            )
            page = browser.new_page(user_agent=HEADERS["User-Agent"])
            page.goto(MSTC_URL, wait_until="networkidle", timeout=45000)
            page.wait_for_timeout(2500)
            html = page.content()
            browser.close()
        return parse_mstc_html(html)
    except Exception:
        return []


def fetch_mstc() -> tuple[list[dict[str, Any]], str]:
    try:
        # Normal secure request first.
        r = requests.get(MSTC_URL, headers=HEADERS, timeout=25, verify=True)
        r.raise_for_status()
    except SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            r = requests.get(MSTC_URL, headers=HEADERS, timeout=25, verify=False)
            r.raise_for_status()
        except requests.RequestException as fallback_exc:
            raise HTTPException(
                status_code=502,
                detail=f"MSTC source fetch failed after TLS verification fallback: {fallback_exc}",
            )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"MSTC source fetch failed: {exc}")

    # The public Market page is client-side rendered. Try the plain HTML first,
    # then use a real Chromium render if the static response contains no table.
    items = parse_mstc_html(r.text)
    if not items:
        items = fetch_mstc_rendered()

    if not items:
        raise HTTPException(
            status_code=502,
            detail="MSTC page was reached, but the market dashboard could not be parsed. The page is client-side rendered; install the Playwright Chromium browser and retry.",
        )
    return items, datetime.now(timezone.utc).isoformat()


def save_snapshot(items: list[dict[str, Any]], fetched_at: str):
    observation_date = datetime.fromisoformat(fetched_at.replace("Z", "+00:00")).date().isoformat()
    with db() as con:
        for item in items:
            con.execute(
                """
                INSERT INTO market_history
                (observation_date, material_key, category, subcategory, price_per_kg, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_date, material_key) DO UPDATE SET
                    category=excluded.category,
                    subcategory=excluded.subcategory,
                    price_per_kg=excluded.price_per_kg,
                    source=excluded.source,
                    fetched_at=excluded.fetched_at
                """,
                (
                    observation_date,
                    item["key"],
                    item.get("category", ""),
                    item.get("subcategory", ""),
                    item["price_per_kg"],
                    MSTC_URL,
                    fetched_at,
                ),
            )
        con.commit()


@app.get("/api/health")
def health():
    return {"ok": True, "service": "Kabadiwala Connect Backend", "mstc_source": MSTC_URL}


@app.get("/api/market/prices")
def market_prices():
    items, fetched_at = fetch_mstc()
    save_snapshot(items, fetched_at)
    return {"source": MSTC_URL, "fetched_at": fetched_at, "items": items}


@app.get("/api/forecast")
def forecast(material: str = Query(..., min_length=1)):
    with db() as con:
        rows = con.execute(
            """
            SELECT observation_date, price_per_kg
            FROM market_history
            WHERE material_key = ?
            ORDER BY observation_date ASC
            """,
            (material,),
        ).fetchall()

    history = [{"date": r["observation_date"], "price_per_kg": r["price_per_kg"]} for r in rows]
    if len(history) < 3:
        return {
            "ready": False,
            "material": material,
            "history_points": len(history),
            "history": history,
            "source": MSTC_URL,
            "message": "At least 3 different daily real observations are required before generating a forecast.",
        }

    x = [[i] for i in range(len(history))]
    y = [float(p["price_per_kg"]) for p in history]
    model = LinearRegression().fit(x, y)
    future_x = [[len(history) + i] for i in range(1, 8)]
    predictions = [max(0.0, float(v)) for v in model.predict(future_x)]
    last_price = y[-1]

    forecast_rows = []
    for i, predicted in enumerate(predictions, start=1):
        pct = ((predicted - last_price) / last_price * 100) if last_price else 0.0
        if pct > 1:
            trend, label = "up", "Rising"
        elif pct < -1:
            trend, label = "down", "Falling"
        else:
            trend, label = "flat", "Stable"
        from datetime import timedelta
        d = datetime.fromisoformat(history[-1]["date"]).date() + timedelta(days=i)
        forecast_rows.append({
            "date": d.isoformat(),
            "predicted_price_per_kg": round(predicted, 2),
            "percent_change": round(pct, 2),
            "trend": trend,
            "trend_label": label,
        })

    final_pct = forecast_rows[-1]["percent_change"]
    if final_pct > 1:
        advisory = "Trend real observations के आधार पर ऊपर है; बेचने का निर्णय local buyer rate और material quality देखकर लें."
    elif final_pct < -1:
        advisory = "Trend real observations के आधार पर नीचे है; बेहतर buyer quote compare करें और quality/quantity factor देखें."
    else:
        advisory = "Trend लगभग stable है; local buyer quotes और material quality compare करके निर्णय लें."

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
