"""Tests for the 5 ungraded extension exercises (0-4). Zero-key."""
import hashlib
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pipeline import config
from pipeline.dataset import (
    _norm, decontaminate,
    fuzzy_decontaminate, ngram_overlap, _ngrams,
)
from pipeline.embed import (
    embed_text, recursive_chunks, content_hash, get_embedder,
    ingest_docs, EMBED_DIM, _hash_embed,
)
from pipeline.kg import (
    extract_triples, extract_triples_llm, build_graph, query,
)
from pipeline.traces import load_traces, flatten, traces_to_bronze
from pipeline.validate import validate


# Extension 0 - Fuzzy decontamination

class TestFuzzyDecontamination:
    def test_exact_still_works(self):
        pairs = [
            {"prompt": "what is the return policy", "chosen": "30 days", "rejected": "no"},
            {"prompt": "how much is shipping", "chosen": "free", "rejected": "$5"},
        ]
        eval_set = [{"input": "what is the return policy", "reference": "30 days"}]
        clean = decontaminate(pairs, eval_set)
        assert len(clean) == 1

    def test_ngram_identical(self):
        s = "Can I return a widget I bought 10 days ago?"
        assert ngram_overlap(s, s) == 1.0

    def test_ngram_disjoint(self):
        assert ngram_overlap("abcdefghijklmno", "zzzzzzzzzzzzzzz") == 0.0

    def test_ngram_catches_paraphrase(self):
        # One word swapped: "bought" -> "purchased" — most 13-grams survive
        a = "Can I return a widget I bought 10 days ago?"
        b = "Can I return a widget purchased 10 days ago?"
        score = ngram_overlap(a, b, n=13)
        assert score > 0.15, f"expected >0.15, got {score:.4f}"

    def test_fuzzy_ngram_drops_paraphrase(self):
        pairs = [
            {"prompt": "Can I return a widget I bought 10 days ago?",
             "chosen": "yes", "rejected": "no"},
            {"prompt": "Can I return a widget purchased 10 days ago?",
             "chosen": "yes", "rejected": "no"},
        ]
        eval_set = [{"input": "Can I return a widget I bought 10 days ago?", "reference": "yes"}]
        clean, dropped = fuzzy_decontaminate(pairs, eval_set, method="ngram", threshold=0.15)
        assert len(clean) == 0, f"expected 0 clean, got {len(clean)}"
        assert len(dropped) == 2

    def test_fuzzy_embed_method(self):
        pairs = [
            {"prompt": "Can I return a widget?", "chosen": "y", "rejected": "n"},
            {"prompt": "something completely unrelated", "chosen": "y", "rejected": "n"},
        ]
        eval_set = [{"input": "Can I return a widget?", "reference": "y"}]
        clean, dropped = fuzzy_decontaminate(pairs, eval_set, method="embed", threshold=0.80)
        assert len(dropped) >= 1


# Extension 1 - Real embeddings / incremental re-embedding

class TestRealEmbeddings:
    def test_hash_embed_stability(self):
        v1 = _hash_embed("returns policy widget")
        v2 = _hash_embed("returns policy widget")
        assert v1 == v2
        assert len(v1) == EMBED_DIM

    def test_get_embedder_defaults_to_hash(self):
        fn, dim = get_embedder()
        assert dim == EMBED_DIM
        assert fn is _hash_embed

    def test_content_hash_stable(self):
        h1 = content_hash("Hello  world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_content_hash_different(self):
        assert content_hash("hello") != content_hash("world")

    def test_incremental_ingestion_skips_unchanged(self):
        rows = ingest_docs(config.DOCS_DIR)
        hashes = {
            f"{r['doc']}#{r['chunk_id']}": r["content_hash"]
            for r in rows if r.get("content_hash")
        }
        rows2 = ingest_docs(config.DOCS_DIR, known_hashes=hashes)
        skipped = [r for r in rows2 if r["embedding"] is None]
        assert len(skipped) == len(rows2), \
            f"expected all {len(rows2)} skipped, got {len(skipped)}"

    def test_embed_text_returns_dim(self):
        v = embed_text("hello world")
        assert len(v) == EMBED_DIM
        assert all(isinstance(x, float) for x in v)


# Extension 2 - LLM KG extraction

class TestLLMKGExtraction:
    def test_mock_llm_extract_returns_triples(self):
        text = config.DOCS_DIR.joinpath("sample.md").read_text()
        triples = extract_triples_llm(text, model="mock")
        assert len(triples) >= 3
        for subj, rel, obj in triples:
            assert isinstance(subj, str) and subj
            assert isinstance(rel, str) and rel
            assert isinstance(obj, str) and obj

    def test_mock_llm_graph_queryable(self):
        text = config.DOCS_DIR.joinpath("sample.md").read_text()
        triples = extract_triples_llm(text, model="mock")
        g = build_graph(triples)
        assert len(g) >= 3
        assert "RETURNABLE_WITHIN" in {r for r, _ in g.get("widget", [])}

    def test_both_extractors_agree(self):
        text = config.DOCS_DIR.joinpath("sample.md").read_text()
        det = extract_triples(text)
        llm = extract_triples_llm(text, model="mock")
        det_subjects = {s for s, _, _ in det}
        llm_subjects = {s for s, _, _ in llm}
        for name in ["widget", "gadget", "sprocket"]:
            assert name in det_subjects and name in llm_subjects


# Extension 3 - Data contract

class TestDataContract:
    def test_yaml_exists(self):
        path = config.ROOT / "datacontract.yaml"
        assert path.is_file()

    def test_covers_three_layers(self):
        text = (config.ROOT / "datacontract.yaml").read_text()
        for layer in ["bronze_orders", "silver_orders", "gold_daily_orders"]:
            assert layer in text

    def test_defines_quality_rules(self):
        text = (config.ROOT / "datacontract.yaml").read_text()
        assert "unique" in text or "not-null" in text
        assert "order_id" in text


# Extension 4 - Backfill safety / idempotency

class TestBackfillSafety:
    def test_main_accepts_date_flag(self):
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, "main.py", "--date", "2026-06-02"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "backfill" in r.stdout.lower()

    def test_main_is_idempotent(self):
        import subprocess, sys
        config.WAREHOUSE.unlink(missing_ok=True)
        config.QUARANTINE.unlink(missing_ok=True)

        r1 = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True, timeout=30,
        )
        assert r1.returncode == 0, f"run1 failed: {r1.stderr}"

        r2 = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True, timeout=30,
        )
        assert r2.returncode == 0, f"run2 failed: {r2.stderr}"
        assert "IDEMPOTENT" in r2.stdout

    def test_row_hash_stable(self):
        from main import _row_hash
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t (a INT, b VARCHAR)")
        con.execute("INSERT INTO t VALUES (1, 'hello'), (2, 'world')")
        h1 = _row_hash(con, "t")
        h2 = _row_hash(con, "t")
        assert h1 == h2
        con.close()

    def test_quarantine_is_idempotent(self):
        from main import build_dag
        config.WAREHOUSE.unlink(missing_ok=True)
        config.QUARANTINE.unlink(missing_ok=True)

        con1 = duckdb.connect(str(config.WAREHOUSE))
        try:
            build_dag(con1).run()
        finally:
            con1.close()
        n_q1 = len(pd.read_csv(config.QUARANTINE, dtype=str))

        con2 = duckdb.connect(str(config.WAREHOUSE))
        try:
            build_dag(con2).run()
        finally:
            con2.close()
        n_q2 = len(pd.read_csv(config.QUARANTINE, dtype=str))
        assert n_q2 == n_q1, f"quarantine grew: {n_q1} -> {n_q2}"
