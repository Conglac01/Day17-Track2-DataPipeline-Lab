"""End-to-end smoke check (zero-key). Exit 0 = a student's setup works.

    python verify.py
"""
import sys
import duckdb

from pipeline import config
from pipeline.streaming import MiniTopic, consume_features
from pipeline.embed import ingest_docs, content_hash
from pipeline.traces import load_traces, traces_to_bronze
from pipeline.dataset import (
    build_eval_set, build_preference_pairs, decontaminate,
    fuzzy_decontaminate, ngram_overlap,
)
from pipeline.features import point_in_time_features, naive_leaky_features
from pipeline.kg import (
    ingest_docs_to_graph, returnable_products, traverse, vector_foil,
    extract_triples_llm,
)
import main


_PASSED = 0
_TOTAL = 0


def check(label, cond):
    global _PASSED, _TOTAL
    _TOTAL += 1
    _PASSED += 1 if cond else 0
    mark = "OK " if cond else "XX "
    print(f"  [{mark}] {label}")
    return cond


def run() -> bool:
    ok = True
    stats = main.main()
    ok &= check("extract loaded raw rows", stats["rows_in"] > 0)
    ok &= check("Silver dropped duplicates (the hook)", stats["dropped_dupes"] >= 1)
    ok &= check("gate quarantined bad records", stats["n_quarantined"] == 3)
    ok &= check("Gold produced daily rows", stats["gold_rows"] >= 1)

    con = duckdb.connect(str(config.WAREHOUSE))
    (dupes,) = con.execute(
        f"SELECT count(*) - count(DISTINCT order_id) FROM {config.SILVER}"
    ).fetchone()
    con.close()
    ok &= check("no duplicate order_id remains in Silver", dupes == 0)

    topic = MiniTopic()
    for i, (k, eid, amt) in enumerate(
        [("u1", "e1", 10), ("u1", "e1", 10), ("u2", "e2", 5)]
    ):
        topic.produce(k, {"event_id": eid, "amount": amt})
    feats = consume_features(topic)
    ok &= check("streaming consumer is idempotent", feats["u1"]["orders"] == 1)

    rows = ingest_docs(config.DOCS_DIR)
    ok &= check("doc->chunk->embedding ingestion", len(rows) > 0)

    # --- Agent-data flywheel (Thực Hành 1/3/4) ---
    fcon = duckdb.connect(":memory:")
    traces = load_traces()
    n_spans = traces_to_bronze(fcon, traces)
    ok &= check("agent traces flattened into Bronze spans", n_spans >= len(traces))
    eval_set = build_eval_set(fcon)
    ok &= check("eval golden set curated from traces", len(eval_set) >= 1)
    pairs = build_preference_pairs(fcon)
    clean = decontaminate(pairs, eval_set)
    ok &= check("decontamination drops eval-overlapping pairs", len(clean) < len(pairs))
    ok &= check("at least one clean preference pair survives", len(clean) >= 1)
    pit = point_in_time_features(fcon)
    leaky = naive_leaky_features(fcon)
    m = pit.merge(leaky, on=["user_id", "event_ts"])
    ok &= check("ASOF point-in-time join avoids future leakage",
                int((m["spend_leaky"] > m["spend_at_event"]).sum()) >= 1)
    fcon.close()

    # --- Knowledge Graph bonus (§13) ---
    graph = ingest_docs_to_graph(config.DOCS_DIR)
    ok &= check("knowledge graph built from docs", len(graph) >= 2)
    rp = returnable_products(graph)
    ok &= check("graph query answers 'what is returnable?' = widget only",
                "widget" in rp and "gadget" not in rp)   # warranty != returnable
    hops = traverse(graph, "widget", "SHIPS_FROM")
    ok &= check("graph answers a real 2-hop question (widget->accessory->warehouse)",
                bool(hops) and hops[0]["hops"] == 2 and "hanoi" in hops[0]["answer"].lower())
    foil = vector_foil(config.DOCS_DIR, "widget", "hanoi")
    ok &= check("vector foil: no single chunk answers the multi-hop question",
                foil["single_chunk_answers_it"] is False)

    # ── Extension exercises (ungraded, for depth) ─────────────────────

    # Ext 0: Fuzzy decontamination
    # A realistic paraphrase attack on exact-match: minimal rewording to evade
    # literal comparison while keeping the same intent.
    paraphrase_pairs = [
        {"prompt": "Can I return a widget I bought 10 days ago?",
         "chosen": "yes", "rejected": "no"},
        {"prompt": "Can I return a widget purchased 10 days ago?",
         "chosen": "yes", "rejected": "no"},
    ]
    paraphrase_eval = [
        {"input": "Can I return a widget I bought 10 days ago?",
         "reference": "yes"},
    ]
    # exact-match: only drops the first
    ok &= check(
        "Ext0: exact decontaminate drops literal match only",
        len(decontaminate(paraphrase_pairs, paraphrase_eval)) == 1,
    )
    # fuzzy 13-gram: drops BOTH — the paraphrase ("bought"→"purchased") is caught
    clean_fuzzy, dropped_ng = fuzzy_decontaminate(
        paraphrase_pairs, paraphrase_eval, method="ngram", threshold=0.15, n=13,
    )
    ok &= check(
        f"Ext0: fuzzy 13-gram decontaminate catches paraphrase "
        f"(dropped {len(dropped_ng)} of 2)",
        len(clean_fuzzy) == 0,
    )

    # Ext 1: Real embeddings — content hash stability & incremental skip
    h1 = content_hash("hello  world")
    h2 = content_hash("hello world")  # whitespace churn ignored
    ok &= check("Ext1: content hash ignores whitespace churn", h1 == h2)
    rows_inc = ingest_docs(config.DOCS_DIR)
    known = {f"{r['doc']}#{r['chunk_id']}": r["content_hash"]
              for r in rows_inc if r.get("content_hash")}
    rows_skip = ingest_docs(config.DOCS_DIR, known_hashes=known)
    skipped = [r for r in rows_skip if r["embedding"] is None]
    ok &= check(
        f"Ext1: incremental ingestion skips unchanged chunks "
        f"({len(skipped)}/{len(rows_skip)})",
        len(skipped) == len(rows_skip),
    )

    # Ext 2: LLM KG extraction
    kg_text = config.DOCS_DIR.joinpath("sample.md").read_text()
    llm_triples = extract_triples_llm(kg_text, model="mock")
    ok &= check(
        f"Ext2: LLM KG extractor returns triples from sample.md "
        f"({len(llm_triples)} triples)",
        len(llm_triples) >= 3,
    )

    # Ext 3: Data contract exists
    dc_path = config.ROOT / "datacontract.yaml"
    ok &= check("Ext3: datacontract.yaml exists", dc_path.is_file())

    # Ext 4: Backfill safety — re-run main.py and assert idempotency
    import subprocess
    r_ext4 = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True, text=True, timeout=30,
    )
    ok &= check(
        "Ext4: main.py re-run is idempotent (IDEMPOTENT ✓ in output)",
        "IDEMPOTENT" in r_ext4.stdout,
    )

    return ok


if __name__ == "__main__":
    print("=== verify.py: Day 17 lab smoke test ===")
    success = run()
    print(f"\nRESULT: {_PASSED}/{_TOTAL} checks — "
          + ("ALL PASS" if success else "FAILURES ABOVE"))
    sys.exit(0 if success else 1)
