"""Database storage. No secrets or external API calls in model code."""
from datetime import datetime, timezone
import time
import numpy as np
import psycopg
from psycopg.types.json import Jsonb

KEYS = 'stock_id,style,analysis_minutes,horizon_minutes'


def connect(dsn):
    conn = psycopg.connect(dsn, sslmode='require', connect_timeout=10, autocommit=True)
    conn.execute("set statement_timeout='15s'")
    return conn


def load_payloads(conn, table):
    # Table identifiers come only from this fixed allowlist, never from requests.
    if table not in ('models', 'calibration', 'latest'):
        raise ValueError('invalid table')
    return {tuple(row[:4]): row[4] for row in conn.execute(
        f'select {KEYS},payload from stock_engine.ml_v2_{table}').fetchall()}


def merge_prices(old, rows, cutoff):
    old_t, old_p = old
    if rows:
        t = np.array([int(r[0].timestamp() // 60) for r in rows], dtype=np.int64)
        p = np.array([float(r[1]) for r in rows])
        times, prices = np.r_[old_t, t], np.r_[old_p, p]
        # Stable sort preserves the existing observation for duplicate timestamps.
        order = np.argsort(times, kind='stable')
        times, prices = times[order], prices[order]
        keep = np.r_[True, np.diff(times) != 0] & (times >= cutoff)
        return times[keep], prices[keep]
    keep = old_t >= cutoff
    return old_t[keep], old_p[keep]


def read_prices(conn, stock, old, watermark):
    query_time = conn.execute('select now()').fetchone()[0]
    if watermark is None:
        rows = conn.execute('''select observed_at,price from stock_engine.prices
            where stock_id=%s and observed_at>=now()-interval '180 days'
            order by observed_at''', (stock,)).fetchall()
    else:
        # Overlap handles normal in-flight collector transactions and backfill.
        rows = conn.execute('''select observed_at,price from stock_engine.prices
            where stock_id=%s and received_at >= %s - interval '2 minutes'
            and observed_at>=now()-interval '180 days' order by observed_at''',
            (stock, watermark)).fetchall()
    cutoff = int(query_time.timestamp() // 60) - 259200
    return merge_prices(old, rows, cutoff), query_time


def save_payloads(conn, table, mapping):
    if table not in ('models', 'calibration'):
        raise ValueError('invalid table')
    timestamp = 'trained_at' if table == 'models' else 'calibrated_at'
    guard = " where coalesce((stock_engine.ml_v2_models.payload->>'trained_through')::bigint,0) <= coalesce((excluded.payload->>'trained_through')::bigint,0)" if table == 'models' else ''
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(f'''insert into stock_engine.ml_v2_{table}
                ({KEYS},payload) values(%s,%s,%s,%s,%s)
                on conflict({KEYS}) do update set payload=excluded.payload,{timestamp}=now(){guard}''',
                [(*key, Jsonb(value)) for key, value in mapping.items()])


def save_results(conn, mapping):
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(f'''insert into stock_engine.ml_v2_latest
                ({KEYS},origin_at,payload) values(%s,%s,%s,%s,%s,%s)
                on conflict({KEYS}) do update set origin_at=excluded.origin_at,
                  payload=excluded.payload,computed_at=now()
                where stock_engine.ml_v2_latest.origin_at<=excluded.origin_at''',
                [(*key, datetime.fromtimestamp(value['origin_minute'] * 60, timezone.utc), Jsonb(value))
                 for key, value in mapping.items()])
            ready = [(key, value) for key, value in mapping.items()
                     if value['status'] == 'ready' and value.get('model_id')
                     and not value.get('forecast_method')]
            # One nonoverlapping issued forecast per key/horizon, not every minute.
            cur.executemany(f'''insert into stock_engine.ml_v2_checks
                ({KEYS},origin_at,target_at,origin_price,raw_return,predicted_return,model_id)
                select %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                where not exists(select 1 from stock_engine.ml_v2_checks
                  where stock_id=%s and style=%s and analysis_minutes=%s and horizon_minutes=%s
                    and target_at>%s)
                on conflict do nothing''',
                [(*key, datetime.fromtimestamp(v['origin_minute']*60, timezone.utc),
                  datetime.fromtimestamp((v['origin_minute']+key[3])*60, timezone.utc),
                  v['origin_price'],v['raw_return'],v['predicted_return'],v['model_id'],
                  *key,datetime.fromtimestamp(v['origin_minute']*60, timezone.utc)) for key,v in ready])


def check_outcomes(conn):
    conn.execute('''update stock_engine.ml_v2_checks f set
        actual_return=(p.price/f.origin_price-1)::double precision,checked_at=now()
        from stock_engine.prices p where f.checked_at is null
          and f.stock_id=p.stock_id and f.target_at=p.observed_at and f.target_at<=now()''')


def health(conn, elapsed, fresh, arrival_age):
    conn.execute('''update stock_engine.ml_v2_health set heartbeat=now(),
        processing_ms=%s,fresh_combinations=%s,missed_latency_target=%s
        where singleton''', (elapsed*1000, fresh, elapsed+arrival_age>10))
