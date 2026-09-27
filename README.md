# Kabadiwala Connect — Backend

This backend is paired with the supplied `index.html` and its existing Forecast screen.

## Run on Mac

```bash
cd backend
chmod +x start_mac.sh
./start_mac.sh
```

Backend: `http://127.0.0.1:8000`

Health check: `http://127.0.0.1:8000/api/health`

Keep the backend terminal running while using the HTML with Live Server.

## Forecast behavior

- `/api/market/prices` fetches MSTC EPRETP market data and stores one observation per material per UTC day in SQLite.
- `/api/forecast?material=<key>` only returns a 7-day ML forecast after at least 3 different daily real observations exist.
- No fake historical values are generated.
- If MSTC changes its page structure or blocks server-side fetching, the frontend displays the real error instead of fabricated prices.

MSTC's public material-market data is used as the source; these values should not be presented as guaranteed local kabadi purchase rates.
