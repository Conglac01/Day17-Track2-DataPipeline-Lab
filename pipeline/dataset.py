"""Thực Hành 4 — Eval & preference datasets curated from agent traces.

Once agent turns live in Bronze (see traces.py), they become training data:

  * golden eval set   — (input -> reference output) from turns that SUCCEEDED.
  * preference pairs   — (prompt, chosen, rejected) where `chosen` is a good
                         answer and `rejected` is a failed/refused answer to the
                         SAME question. This is exactly the (x, y_w, y_l) format
                         DPO/ORPO consume on Day 22.

The decontamination step is the part students always skip and regret: any prompt
that appears in the eval set is removed from the preference (training) set, so we
never train on what we grade on. Train/test leakage is the #1 way an eval lies.
"""
from __future__ import annotations
import json
from pathlib import Path

import duckdb

from .traces import BRONZE_SPANS


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def build_eval_set(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Golden eval rows = the human-curated holdout (split='eval') of
    successful turns: {input, reference, trace_id}. Kept OUT of training."""
    rows = con.execute(
        f"""
        SELECT trace_id, user_input, agent_output
        FROM {BRONZE_SPANS}
        WHERE depth = 0 AND status = 'ok' AND split = 'eval'
              AND user_input IS NOT NULL AND agent_output IS NOT NULL
        ORDER BY trace_id
        """
    ).fetchall()
    return [
        {"trace_id": t, "input": i, "reference": o}
        for (t, i, o) in rows
    ]


def build_preference_pairs(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Pair a good (ok) answer with a bad (error) answer to the SAME question.

    `chosen`  = output of a successful turn for that input.
    `rejected` = output of a failed/refused turn for that same input.
    Yields the (prompt, chosen, rejected) triple DPO/ORPO expect (Day 22).
    """
    roots = con.execute(
        f"""
        SELECT user_input, status, agent_output
        FROM {BRONZE_SPANS}
        WHERE depth = 0 AND user_input IS NOT NULL
        """
    ).fetchall()

    good: dict[str, str] = {}
    bad: dict[str, str] = {}
    for user_input, status, output in roots:
        key = _norm(user_input)
        if status == "ok":
            good.setdefault(key, output)
        else:
            bad.setdefault(key, output)

    pairs = []
    for key in sorted(set(good) & set(bad)):
        pairs.append({"prompt": key, "chosen": good[key], "rejected": bad[key]})
    return pairs


def decontaminate(pairs: list[dict], eval_set: list[dict]) -> list[dict]:
    """Drop any preference pair whose prompt is also an eval input (no leakage).

    NOTE: this is EXACT-match decontamination (after lowercase + whitespace
    normalization). A reworded or paraphrased duplicate still leaks — production
    pipelines add n-gram (e.g. 13-gram) or embedding-similarity matching. See the
    'fuzzy decontamination' extension exercise.
    """
    held_out = {_norm(e["input"]) for e in eval_set}
    return [p for p in pairs if _norm(p["prompt"]) not in held_out]


# ═══════════════════════════════════════════════════════════════════════
# Extension 0 — Fuzzy decontamination: n-gram overlap + embedding cosine
# ═══════════════════════════════════════════════════════════════════════

def _ngrams(text: str, n: int = 13) -> set[str]:
    """Character n-grams: robust to small rewrites, word-order changes."""
    t = _norm(text)
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def ngram_overlap(a: str, b: str, n: int = 13) -> float:
    """Jaccard similarity on character n-grams. Returns 0.0–1.0."""
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def fuzzy_decontaminate(
    pairs: list[dict],
    eval_set: list[dict],
    method: str = "ngram",
    threshold: float = 0.30,
    n: int = 13,
) -> tuple[list[dict], list[dict]]:
    """Drop preference pairs whose prompt is *too similar* to any eval input.

    Two methods (selectable):
      * ``ngram``  — character 13-gram Jaccard. Fast, language-agnostic.
      * ``embed``   — cosine on embedding vectors (requires embed_text from embed.py).

    Returns (clean_pairs, dropped_pairs) so you can inspect what was removed.
    """
    eval_inputs = [e["input"] for e in eval_set]

    if method == "embed":
        from pipeline.embed import embed_text as _embed

        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = (sum(x * x for x in a)) ** 0.5
            nb = (sum(y * y for y in b)) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        eval_vecs = [_embed(t) for t in eval_inputs]
        clean, dropped = [], []
        for p in pairs:
            pvec = _embed(p["prompt"])
            if any(_cosine(pvec, ev) >= threshold for ev in eval_vecs):
                dropped.append(p)
            else:
                clean.append(p)
    else:  # ngram
        clean, dropped = [], []
        for p in pairs:
            if any(
                ngram_overlap(p["prompt"], ei, n=n) >= threshold for ei in eval_inputs
            ):
                dropped.append(p)
            else:
                clean.append(p)

    return clean, dropped


def write_jsonl(rows: list[dict], path: Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
