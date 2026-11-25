"""Provision the private backtest tables on every configured stock shard."""
from __future__ import annotations
import json
import os
from pathlib import Path
import psycopg

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "python" / "stock_ml" / "backtest_setup.sql"


def statements(sql: str) -> list[str]:
    # The schema file contains plain DDL only (no procedural blocks).
    return [part.strip() for part in sql.split(";") if part.strip()]


def provision(projects: dict[str, str]) -> None:
    sql = SCHEMA.read_text(encoding="utf-8")
    commands = [statement for statement in statements(sql)
                if statement.strip().lower() not in {'begin', 'commit'}]
    for project, dsn in projects.items():
        number = int(project)
        if not 1 <= number <= 18:
            raise ValueError(f"Invalid shard number: {number}")
        with psycopg.connect(dsn, sslmode="require", connect_timeout=15) as conn:
            settings = conn.execute(
                "select project_number, enabled from stock_engine.settings where singleton"
            ).fetchone()
            if not settings or int(settings[0]) != number or not settings[1]:
                raise ValueError(f"Shard {number}: project assignment mismatch or disabled")
            with conn.transaction():
                for command in commands:
                    conn.execute(command)
        # Verify the table used by the ranking step is now present.
        with psycopg.connect(dsn, sslmode="require", connect_timeout=15) as conn:
            exists = conn.execute(
                "select to_regclass('stock_engine.ml_backtest_reports')"
            ).fetchone()[0]
            if exists is None:
                raise RuntimeError(f"Shard {number}: ml_backtest_reports was not created")
        print(json.dumps({"event": "backtest_schema_ready", "project": number,
                          "table": "stock_engine.ml_backtest_reports"}))


def main() -> int:
    try:
        projects = json.loads(os.getenv("STOCK_ML_PROJECTS_JSON", "{}"))
    except ValueError as exc:
        raise SystemExit("STOCK_ML_PROJECTS_JSON must be valid JSON") from exc
    if not isinstance(projects, dict) or not projects:
        raise SystemExit("Configure STOCK_ML_PROJECTS_JSON in repository secrets")
    provision(projects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
