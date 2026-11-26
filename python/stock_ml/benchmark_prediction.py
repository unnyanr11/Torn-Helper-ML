import json, os, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import psycopg
from algorithm_registry import ALGORITHM_REGISTRY, family_for_algorithm_id
from backtesting import fit_live_model
from clean_series import reconstruct_short_gaps
from realtime_model import COMBINATIONS, feature_matrix, infer_model, prepare_series

ASSIGNMENTS = {i: i for i in range(1, 7)}

def run_stock(project, dsn, stock, lookback, max_rows):
    started = time.perf_counter()
    family = family_for_algorithm_id(ASSIGNMENTS[stock])
    with psycopg.connect(dsn, sslmode="require", connect_timeout=20) as conn:
        setting = conn.execute("select project_number,enabled from stock_engine.settings where singleton").fetchone()
        if not setting or int(setting[0]) != project or not setting[1]:
            raise RuntimeError(f"Shard {project} is disabled or misconfigured")
        rows = conn.execute("""select floor(extract(epoch from observed_at)/60)::bigint,
            (array_agg(price order by observed_at desc))[1]
            from stock_engine.prices where stock_id=%s
            and observed_at>=now()-(%s * interval '1 day')
            group by 1 order by 1""", (stock, lookback)).fetchall()
    fetch_s = time.perf_counter() - started
    if len(rows) < 1000:
        raise RuntimeError(f"Stock {stock}: insufficient price history ({len(rows)} rows)")
    times = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
    prices = np.asarray([float(r[1]) for r in rows], dtype=float)
    prep = time.perf_counter()
    times, prices, synthetic, _ = reconstruct_short_gaps(times, prices)
    observed = ~synthetic
    prepared = prepare_series(times, prices)
    origin = int(times[-1])
    prep_s = time.perf_counter() - prep
    fit_s = infer_s = 0.0
    predicted = 0
    skipped = Counter()
    for style, window, horizon in COMBINATIONS:
        t0 = time.perf_counter()
        try:
            model = fit_live_model(times, prices, window_minutes=window,
                horizon_minutes=horizon, model_family=family,
                observed_mask=observed, prepared=prepared,
                max_training_rows=max_rows)
        except ValueError as exc:
            fit_s += time.perf_counter() - t0
            skipped[str(exc)] += 1
            continue
        elapsed = time.perf_counter() - t0
        fit_s += elapsed
        t1 = time.perf_counter()
        x, valid = feature_matrix(times, prices, np.asarray([origin]), window, prepared)
        if not bool(valid[0]):
            skipped["latest_feature_window_incomplete"] += 1
            continue
        value = float(infer_model(model, x[0]))
        if not np.isfinite(value):
            raise RuntimeError("Non-finite prediction")
        infer_s += time.perf_counter() - t1
        predicted += 1
    return {
        "stock_id": stock, "algorithm_id": ASSIGNMENTS[stock], "model_family": family,
        "raw_observations": len(rows), "usable_observations": len(times),
        "selectors_total": len(COMBINATIONS), "selectors_predicted": predicted,
        "selectors_skipped": len(COMBINATIONS)-predicted, "skip_reasons": dict(skipped),
        "fetch_seconds": round(fetch_s, 3), "preprocessing_seconds": round(prep_s, 3),
        "fit_seconds": round(fit_s, 3), "inference_seconds": round(infer_s, 3),
        "stock_total_seconds": round(time.perf_counter()-started, 3)
    }

def main():
    lookback = int(os.getenv("BENCHMARK_LOOKBACK_DAYS", "180"))
    max_rows = int(os.getenv("BENCHMARK_MAX_TRAINING_ROWS", "3000"))
    if not 30 <= lookback <= 180 or not 240 <= max_rows <= 10000:
        raise SystemExit("Invalid benchmark lookback or training row cap")
    projects = {int(k): v for k, v in json.loads(os.environ["STOCK_ML_PROJECTS_JSON"]).items()}
    workers = int(os.getenv("BENCHMARK_WORKERS", "2"))
    if not 1 <= workers <= 6:
        raise SystemExit("BENCHMARK_WORKERS must be between 1 and 6")
    results, failures = [], []
    started = time.perf_counter()
    print(json.dumps({"event": "benchmark_started", "workers": workers,
                      "stocks": 6, "selectors_per_stock": len(COMBINATIONS)}), flush=True)
    # Each process owns its DB connection and stock arrays/models. Collect
    # results as stocks finish; estimator-level native threads remain capped at 1.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_stock, (stock + 1) // 2, projects[(stock + 1) // 2],
                        stock, lookback, max_rows): stock
            for stock in range(1, 7)
        }
        for future in as_completed(futures):
            stock = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({"event": "stock_benchmark", **result}), flush=True)
            except Exception as exc:
                failure = {"stock_id": stock, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(json.dumps({"event": "stock_benchmark_failed", **failure}), flush=True)
    results.sort(key=lambda row: row["stock_id"])
    failures.sort(key=lambda row: row["stock_id"])
    elapsed = time.perf_counter()-started
    summary = {
        "lookback_days": lookback, "max_training_rows_per_selector": max_rows,
        "worker_processes": workers,
        "algorithm_assignments": {str(k): {"algorithm_id": k, "family": ALGORITHM_REGISTRY[k]} for k in ASSIGNMENTS.values()},
        "stocks_completed": len(results), "failures": failures,
        "selectors_predicted": sum(r["selectors_predicted"] for r in results),
        "selectors_skipped": sum(r["selectors_skipped"] for r in results),
        "total_wall_clock_seconds": round(elapsed, 3), "stocks": results
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered, flush=True)
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write("# Six-stock Python prediction benchmark\n\n")
            out.write("| Stock | Algorithm | Predictions | Skipped | Fit seconds | Inference seconds | Total seconds |\n|---:|---|---:|---:|---:|---:|---:|\n")
            for r in results:
                out.write(f"| {r['stock_id']} | {r['algorithm_id']} — {r['model_family']} | {r['selectors_predicted']}/{r['selectors_total']} | {r['selectors_skipped']} | {r['fit_seconds']:.3f} | {r['inference_seconds']:.3f} | {r['stock_total_seconds']:.3f} |\n")
            out.write(f"\n**Worker processes:** {workers}\n\n**Total wall-clock:** {elapsed:.3f} seconds\n\n**Completed stocks:** {len(results)}/6\n")
            if failures:
                out.write("\n## Failures\n")
                for f in failures:
                    out.write(f"\n- Stock {f['stock_id']}: {f['error']}")
            out.write("\n\n<details><summary>Full JSON</summary>\n\n\x60\x60\x60json\n" + rendered + "\n\x60\x60\x60\n</details>\n")
    if failures or not results or not summary["selectors_predicted"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
