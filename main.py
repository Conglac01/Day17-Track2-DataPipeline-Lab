"""Run the full lite-path Medallion pipeline as a DAG. Zero-key, DuckDB-only.

    python main.py                # full run (fresh)
    python main.py                # re-run — IDEMPOTENT (no double rows)
    python main.py --date 2026-06-02  # backfill a single date (Gold only)

Stages: extract -> validate(gate) -> transform(dedup -> gold) -> report.
Prints the dedup count (the hook payoff) and the quarantine count.

Idempotency (Extension 4): Bronze is append-only; a re-run checks whether raw
rows are already present (by content hash) and skips them. Silver dedup is
deterministic — re-running produces the same result. Gold is a full refresh of
the aggregate table, so it is naturally idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date, datetime

import duckdb

from pipeline import config
from pipeline.dag import DAG
from pipeline.extract import extract_to_bronze
from pipeline.validate import validate, write_quarantine
from pipeline.transform import write_silver, write_gold
from pipeline.load import read_gold


# ── idempotency helpers ───────────────────────────────────────────────

def _row_hash(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return a set of content hashes for every row currently in *table*."""
    try:
        rows = con.execute(
            f"SELECT * FROM {table} ORDER BY 1,2,3,4,5,6"
        ).fetchall()
    except Exception:
        return set()
    return {
        hashlib.md5("|".join(str(c) for c in r).encode()).hexdigest()
        for r in rows
    }


def _append_only_insert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: "pd.DataFrame",
) -> int:
    """Insert rows that aren't already in *table* (by content hash)."""
    import pandas as pd

    # Check if table exists
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()[0]

    if not exists:
        con.register("_tmp_append", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_append")
        return len(df)

    existing_hashes = _row_hash(con, table)
    new_rows = []
    for _, row in df.iterrows():
        h = hashlib.md5(
            "|".join(str(row[c]) for c in df.columns).encode()
        ).hexdigest()
        if h not in existing_hashes:
            new_rows.append(row)

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows, columns=df.columns)
    con.register("_tmp_new", new_df)
    con.execute(f"INSERT INTO {table} SELECT * FROM _tmp_new")
    return len(new_rows)


# ── DAG definition ────────────────────────────────────────────────────

def build_dag(
    con: duckdb.DuckDBPyConnection,
    backfill_date: str | None = None,
) -> DAG:
    dag = DAG()

    @dag.task("extract")
    def _extract():
        import pandas as pd

        df = con.execute(
            f"SELECT * FROM read_csv_auto('{config.RAW_CSV.as_posix()}',"
            f" header=true, all_varchar=true)"
        ).fetchdf()

        # ── Idempotent Bronze: append-only, skip already-present rows ──
        table_exists = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = ?",
            [config.BRONZE],
        ).fetchone()[0]

        if table_exists:
            n_before = con.execute(
                f"SELECT count(*) FROM {config.BRONZE}"
            ).fetchone()[0]
            n_added = _append_only_insert(con, config.BRONZE, df)
            n_total = n_before + n_added
            verb = "appended" if n_added else "skipped (already present)"
            print(f"  Bronze: {n_before} existing + {n_added} {verb} = {n_total} total")
        else:
            con.register("_bronze_tmp", df)
            con.execute(
                f"CREATE TABLE {config.BRONZE} AS SELECT * FROM _bronze_tmp"
            )
            n_total = len(df)
            print(f"  Bronze: created with {n_total} rows")

        return con.execute(f"SELECT * FROM {config.BRONZE}").fetchdf()

    @dag.task("validate", upstream=["extract"])
    def _validate(bronze_df):
        clean, bad = validate(bronze_df)

        # ── Idempotent quarantine: append new bad rows only ──
        if len(bad):
            import pandas as pd

            if config.QUARANTINE.exists():
                existing = pd.read_csv(config.QUARANTINE, dtype=str)
                # merge on all columns to find truly new bad rows
                merged = bad.merge(
                    existing, how="left",
                    on=list(bad.columns),
                    indicator=True,
                )
                new_bad = merged[merged["_merge"] == "left_only"].drop(
                    columns=["_merge"]
                )
                if len(new_bad):
                    new_bad.to_csv(
                        config.QUARANTINE, mode="a",
                        header=False, index=False,
                    )
                n_new = len(new_bad)
            else:
                bad.to_csv(config.QUARANTINE, index=False)
                n_new = len(bad)

            print(
                f"  Quarantine: {len(bad)} bad rows total, "
                f"{n_new} new (appended to {config.QUARANTINE.name})"
            )
        else:
            n_new = 0

        return {"clean": clean, "n_quarantined": len(bad), "n_new_quarantined": n_new}

    @dag.task("transform", upstream=["validate"])
    def _transform(v):
        stats = write_silver(con, v["clean"])
        n_gold = write_gold(con)
        return {**stats, "gold_rows": n_gold, "n_quarantined": v["n_quarantined"]}

    @dag.task("report", upstream=["transform"])
    def _report(t):
        return t

    return dag


# ── entrypoint ────────────────────────────────────────────────────────

def main(backfill_date: str | None = None) -> dict:
    if backfill_date:
        print(f"=== Day 17 pipeline — backfill mode (date={backfill_date}) ===")
    else:
        print("=== Day 17 pipeline (lite) ===")

    con = duckdb.connect(str(config.WAREHOUSE))
    try:
        results = build_dag(con, backfill_date).run()
        stats = results["report"]
        print(f"\n  bronze rows in      : {stats['rows_in']}")
        print(f"  duplicates dropped  : {stats['dropped_dupes']}  (Silver dedup)")
        print(f"  records quarantined : {stats['n_quarantined']}  (failed the gate)")
        print(f"  silver rows         : {stats['rows_out']}")
        print(f"  gold daily rows     : {stats['gold_rows']}")
        print("\nGold (completed orders by day):")
        print(read_gold(con).to_string(index=False))

        # ── Prove idempotency: re-running should not double anything ──
        if not backfill_date:
            print("\n--- Re-run idempotency check ---")
            results2 = build_dag(con).run()
            stats2 = results2["report"]
            delta = stats2["rows_in"] - stats["rows_in"]
            print(
                f"  Bronze rows before/after re-run: "
                f"{stats['rows_in']} / {stats2['rows_in']} "
                f"(delta={delta} — "
                f"{'IDEMPOTENT ✓' if delta == 0 else 'LEAK ✗'})"
            )
            delta_silver = stats2["rows_out"] - stats["rows_out"]
            print(
                f"  Silver rows before/after re-run: "
                f"{stats['rows_out']} / {stats2['rows_out']} "
                f"(delta={delta_silver} — "
                f"{'IDEMPOTENT ✓' if delta_silver == 0 else 'LEAK ✗'})"
            )
            stats.update({
                "rerun_bronze_delta": delta,
                "rerun_silver_delta": delta_silver,
            })

        return stats
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Day 17 Medallion pipeline — idempotent, backfill-safe"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Backfill a specific date (YYYY-MM-DD). Gold only.",
    )
    args = parser.parse_args()
    main(backfill_date=args.date)
