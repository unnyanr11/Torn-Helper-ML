"""Backfill real one-minute Torn stock prices into their assigned SG shard.

Uses Tornsy's public historical minute endpoint. It only inserts observations
that are absent; live rows are never overwritten and no data is sent to the
main application database.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import argparse
import json
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psycopg

# Canonical Torn stock ID -> Tornsy acronym.
TICKERS = {
    1: "TSB", 2: "TCI", 3: "SYS", 4: "LAG", 5: "IOU", 6: "GRN",
    7: "THS", 8: "YAZ", 9: "TCT", 10: "CNC", 11: "MSG", 12: "TMI",
    13: "TCP", 14: "IIL", 15: "FHG", 16: "SYM", 17: "LSC", 18: "PRN",
    19: "EWM", 20: "TCM", 21: "ELT", 22: "HRG", 23: "TGP", 24: "MUN",
    25: "WSU", 26: "IST", 27: "BAG", 28: "EVL", 29: "MCS", 30: "WLT",
    31: "TCC", 32: "ASS", 33: "CBD", 34: "LOS", 35: "PTS",
}
CHUNK_MINUTES = 1800
PAGE_LIMIT = 2000
RATE_INTERVAL = 0.25  # Four public API requests per second across all shards.
_rate_lock = threading.Lock()
_next_request = 0.0


def assigned_stock_ids(project):
    if not 1 <= int(project) <= 18:
        raise ValueError("Project number must be between 1 and 18")
    return [int(project) * 2 - 1, int(project) * 2] if int(project) < 18 else [35]


def rate_limit():
    global _next_request
    with _rate_lock:
        now = time.monotonic()
        wait = max(0.0, _next_request - now)
        _next_request = max(now, _next_request) + RATE_INTERVAL
    if wait:
        time.sleep(wait)


def minute_chunks(start_ts, end_ts, chunk_minutes=CHUNK_MINUTES):
    """Yield non-overlapping [from, to) windows, aligned to minute boundaries."""
    start = (int(start_ts) // 60) * 60
    end = (int(end_ts) // 60) * 60
    cursor = start
    step = chunk_minutes * 60
    while cursor < end:
        nxt = min(end, cursor + step)
        yield cursor, nxt
        cursor = nxt


def parse_rows(payload, start_ts, end_ts):
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Tornsy response has no data array")
    rows = {}
    for item in payload["data"]:
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            stamp = int(item[0])
            price = float(item[1])
            shares = int(item[2]) if len(item) > 2 and item[2] is not None else None
        except (TypeError, ValueError, OverflowError):
            continue
        if start_ts <= stamp < end_ts and stamp % 60 == 0 and price > 0:
            rows[stamp] = (price, shares)
    return [(stamp, *rows[stamp]) for stamp in sorted(rows)]


def fetch_chunk(ticker, start_ts, end_ts, opener=urlopen, limiter=rate_limit, sleep=time.sleep):
    url = "https://tornsy.com/api/" + ticker + "?" + urlencode({
        "interval": "m1", "from": start_ts, "to": end_ts, "limit": PAGE_LIMIT,
    })
    last_error = None
    for attempt in range(6):
        limiter()
        try:
            request = Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "Torn-Helper-ML-minute-backfill/1.0",
            })
            with opener(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return parse_rows(payload, start_ts, end_ts)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After")
            if not retry_after:
                delay = min(30.0, 2 ** attempt)
            else:
                try:
                    delay = min(60.0, max(1.0, float(retry_after)))
                except (TypeError, ValueError):
                    try:
                        delay = min(60.0, max(1.0, (parsedate_to_datetime(retry_after) -
                            datetime.now(timezone.utc)).total_seconds()))
                    except Exception:
                        delay = min(30.0, 2 ** attempt)
            sleep(delay)
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt == 5:
                break
            sleep(min(30.0, 2 ** attempt))
    raise RuntimeError(f"Tornsy request failed after retries: {type(last_error).__name__}") from None


def insert_chunk(conn, stock_id, rows, columns):
    if not rows:
        return 0
    names = ["stock_id", "observed_at", "price"]
    if "total_shares" in columns:
        names.append("total_shares")
    if "source" in columns:
        names.append("source")
    quoted = ", ".join(f'"{name}"' for name in names)
    placeholders = ", ".join(["%s"] * len(names))
    sql = f"insert into stock_engine.prices ({quoted}) values ({placeholders}) on conflict do nothing"
    values = []
    for stamp, price, shares in rows:
        row = [stock_id, datetime.fromtimestamp(stamp, timezone.utc), price]
        if "total_shares" in columns:
            row.append(shares)
        if "source" in columns:
            row.append("Tornsy")
        values.append(tuple(row))
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(sql, values)
            return max(0, cur.rowcount)


def backfill_project(project, dsn, days, dry_run=False):
    expected = assigned_stock_ids(project)
    with psycopg.connect(dsn, sslmode="require", connect_timeout=15, autocommit=True) as conn:
        settings = conn.execute(
            "select project_number, enabled from stock_engine.settings where singleton"
        ).fetchone()
        if not settings or int(settings[0]) != project:
            raise ValueError(f"SG{project}: project assignment mismatch")
        if not settings[1]:
            raise ValueError(f"SG{project}: shard is disabled")
        assigned = [int(row[0]) for row in conn.execute(
            "select stock_id from stock_engine.stocks order by stock_id"
        ).fetchall()]
        if assigned != expected:
            raise ValueError(f"SG{project}: expected stock IDs {expected}, found {assigned}")
        metadata = conn.execute("""
            select column_name, is_nullable, column_default
            from information_schema.columns
            where table_schema='stock_engine' and table_name='prices'
        """).fetchall()
        columns = {row[0] for row in metadata}
        required = {"stock_id", "observed_at", "price"}
        if not required.issubset(columns):
            raise ValueError(f"SG{project}: stock_engine.prices missing required columns")
        supported = required | {"total_shares", "source"}
        unsupported_required = [
            row[0] for row in metadata
            if row[0] not in supported and row[1] == "NO" and row[2] is None
        ]
        if unsupported_required:
            raise ValueError(f"SG{project}: unsupported required price columns {unsupported_required}")

        end_ts = (int(datetime.now(timezone.utc).timestamp()) // 60) * 60
        start_ts = end_ts - days * 86400
        summary = {"project": project, "days": days, "stocks": {}}
        for stock_id in assigned:
            ticker = TICKERS[stock_id]
            requested = inserted = 0
            for start, end in minute_chunks(start_ts, end_ts):
                rows = fetch_chunk(ticker, start, end)
                requested += len(rows)
                if not dry_run:
                    inserted += insert_chunk(conn, stock_id, rows, columns)
            summary["stocks"][str(stock_id)] = {
                "ticker": ticker, "fetched": requested, "inserted": inserted,
            }
            print(json.dumps({"event": "stock_backfill_complete", "project": project,
                "stock_id": stock_id, "ticker": ticker, "fetched": requested,
                "inserted": inserted, "dry_run": dry_run}), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.days <= 180:
        parser.error("--days must be between 1 and 180 (the ML retention window)")
    try:
        projects = json.loads(os.environ.get("STOCK_ML_PROJECTS_JSON", "{}"))
        if set(map(int, projects)) != set(range(1, 19)):
            raise ValueError()
    except (TypeError, ValueError):
        parser.error("STOCK_ML_PROJECTS_JSON must contain all shard keys 1 through 18")
    failed = []
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(backfill_project, int(project), dsn, args.days, args.dry_run): int(project)
            for project, dsn in projects.items()
        }
        for future in as_completed(futures):
            project = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failed.append(project)
                # Never print DSNs or provider response bodies.
                print(json.dumps({"event": "shard_backfill_failed", "project": project,
                    "error": type(exc).__name__, "message": str(exc)[:240]}), flush=True)
    print(json.dumps({"event": "backfill_summary", "days": args.days,
        "dry_run": args.dry_run, "completed_shards": sorted(r["project"] for r in results),
        "failed_shards": sorted(failed)}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
