"""Bonus (§13) — Knowledge Graph construction from unstructured docs.

Vector RAG retrieves *chunks*; a knowledge graph retrieves *connected facts*,
which is what wins multi-hop and global-summary questions (GraphRAG-Bench,
ICLR'26). The construction pipeline is the same shape regardless of extractor:

    text -> extract (entity, relation, entity) triples -> upsert into a graph

Here the extractor is a small **deterministic** rule matcher so the lab stays
zero-key and reproducible. In production this node is an LLM (e.g. LangChain
`LLMGraphTransformer`) with entity resolution — swap `extract_triples` and the
rest of the pipeline is unchanged. That substitutability is the lesson.
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path

_PRODUCTS = ["widget", "gadget", "sprocket"]

# Entity resolution: collapse surface forms (plural, the/a) to one canonical node
# so an IS_A object ("accessory") matches the SHIPS_FROM subject ("accessories").
# A real pipeline does this with an LLM or a fuzzy matcher; here it is a small map.
_CANON = {
    "widgets": "widget", "gadgets": "gadget", "sprockets": "sprocket",
    "accessories": "accessory", "industrial-parts": "industrial-part",
}


def _canon(entity: str) -> str:
    e = entity.strip().strip(".").lower()
    e = re.sub(r"^(the|a|an)\s+", "", e)
    return _CANON.get(e, e)


# ═══════════════════════════════════════════════════════════════════════
# Extension 2 — LLM-based KG extraction (swappable extractor)
# ═══════════════════════════════════════════════════════════════════════


def extract_triples_llm(
    text: str,
    *,
    model: str = "mock",
    api_key: str | None = None,
) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) triples using an LLM.

    This is the **LLM slot** that replaces ``extract_triples`` in production.
    The downstream graph code (``build_graph``, ``query``, ``traverse``) is unchanged
    — only the extractor changes, demonstrating substitutability.

    ``model="mock"`` runs a small local simulation (no API key needed) that exercises
    the same interface. Pass a real model name + key for production use.

    Prompt design follows the deck's entity-relation extraction pattern:
    entity resolution happens inside the LLM call by asking for canonical forms.
    """
    if model == "mock":
        # Simulated LLM response — exercises the exact same interface without a key.
        # In production this body is replaced by an API call (OpenAI, Anthropic, etc.)
        return _mock_llm_extract(text)

    # Real LLM path (requires an API key)
    import json as _json

    prompt = (
        "Extract (subject, relation, object) knowledge triples from the text below.\n"
        "Rules:\n"
        "- Canonicalize entities: collapse plurals, strip articles (the/a/an).\n"
        "- Use relation names: IS_A, SHIPS_FROM, RETURNABLE_WITHIN, HAS_WARRANTY, "
        "NON_RETURNABLE, HAS_PRICE, BELONGS_TO.\n"
        "- Output ONLY a JSON array of [subject, relation, object] arrays.\n"
        "- No extra text, no explanation.\n\n"
        f"Text:\n{text}\n\nOutput:"
    )

    # --- Replace the try/except body with the LLM SDK of your choice ---
    # Example for anthropic / openai:
    #
    #   import anthropic
    #   client = anthropic.Anthropic(api_key=api_key)
    #   resp = client.messages.create(
    #       model=model, max_tokens=1024,
    #       messages=[{"role": "user", "content": prompt}],
    #   )
    #   raw = resp.content[0].text
    #
    # Example for a local model (Ollama):
    #
    #   import requests
    #   resp = requests.post("http://localhost:11434/api/generate", json={
    #       "model": model, "prompt": prompt, "stream": False,
    #   })
    #   raw = resp.json()["response"]
    #
    # For now we fall back to mock — the pipeline shape is the point.
    try:
        raise NotImplementedError(
            "Set model='mock' or wire a real LLM SDK call here."
        )
    except NotImplementedError:
        return _mock_llm_extract(text)


def _mock_llm_extract(text: str) -> list[tuple[str, str, str]]:
    """Simulate an LLM call: call the deterministic extractor + canonicalize output.

    This is a placeholder — in production an LLM would handle far more varied
    surface forms. The point is to demonstrate the **pipeline shape**:
    swap the extractor, leave the graph untouched.
    """
    raw = extract_triples(text)
    # Run entity resolution on both subject and object
    canon: list[tuple[str, str, str]] = []
    for subj, rel, obj in raw:
        cs = _canon(subj)
        co = _canon(obj) if not obj[0].isdigit() and " " not in obj.split("-", 1)[0] else obj
        canon.append((cs, rel, co))
    # Dedup: LLM might emit duplicate triples from different sentences
    seen = set()
    deduped: list[tuple[str, str, str]] = []
    for t in canon:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Deterministic (subject, relation, object) extraction. LLM slots in here.

    Produces two kinds of edge:
      * entity -> literal  (RETURNABLE_WITHIN, HAS_WARRANTY, NON_RETURNABLE)
      * entity -> entity   (IS_A, SHIPS_FROM) -- these are what make the graph
        *traversable*, so a multi-hop question (widget -> accessory -> warehouse)
        can be answered by walking edges.

    A real pipeline swaps this body for an LLM (e.g. LangChain
    ``LLMGraphTransformer``) plus entity resolution; the downstream graph code is
    unchanged. That substitutability is the lesson.
    """
    triples: list[tuple[str, str, str]] = []
    sentences: list[str] = []
    for line in text.splitlines():          # skip markdown headings + split per line
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    for sentence in sentences:
        low = sentence.lower()

        # per-product policy edges (entity -> literal)
        for prod in _PRODUCTS:
            if prod not in low:
                continue
            ret = re.search(r"within\s+(\d+)\s*-?\s*day", low)
            if "return" in low and ret:
                triples.append((prod, "RETURNABLE_WITHIN", f"{ret.group(1)} days"))
            war = re.search(r"(\d+)\s*-?\s*day", low)
            if "warranty" in low and war:
                triples.append((prod, "HAS_WARRANTY", f"{war.group(1)} days"))
            if "final sale" in low or "cannot be returned" in low:
                triples.append((prod, "NON_RETURNABLE", "final sale"))

        # category membership (entity -> entity): "X and Y belong to the Z category"
        m = re.search(r"^(.*?) belong to the ([\w-]+) category", low)
        if m:
            cat = _canon(m.group(2))
            for subj in re.split(r",|\band\b", m.group(1)):
                subj = _canon(subj)
                if subj:
                    triples.append((subj, "IS_A", cat))

        # logistics (entity -> entity): "Z ship from the <place>"
        m = re.search(r"^([\w-]+) ship from the (.+?)[.]?$", low)
        if m:
            triples.append((_canon(m.group(1)), "SHIPS_FROM", m.group(2).strip()))

    return triples


def build_graph(triples: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Adjacency list: node -> [(relation, neighbour), ...]. Pure-Python, no deps."""
    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subj, rel, obj in triples:
        graph[subj].append((rel, obj))
    return dict(graph)


def ingest_docs_to_graph(docs_dir: Path) -> dict:
    triples: list[tuple[str, str, str]] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        triples.extend(extract_triples(path.read_text(encoding="utf-8")))
    return build_graph(triples)


def query(graph: dict, subject: str, relation: str | None = None) -> list[tuple[str, str]]:
    """1-hop lookup: facts about `subject`, optionally filtered by relation."""
    edges = graph.get(subject, [])
    return [(r, o) for (r, o) in edges if relation is None or r == relation]


def returnable_products(graph: dict) -> list[str]:
    """Connected-fact query a flat vector lookup struggles with: 'what is returnable?'

    A product is returnable iff it has a RETURNABLE_WITHIN edge and is NOT marked
    final-sale. Note a warranty (HAS_WARRANTY) does NOT make something returnable —
    conflating the two is exactly the kind of relation error a graph lets you avoid.
    """
    out = []
    for node, edges in graph.items():
        rels = {r for r, _ in edges}
        if "RETURNABLE_WITHIN" in rels and "NON_RETURNABLE" not in rels:
            out.append(node)
    return sorted(out)


def traverse(graph: dict, start: str, target_relation: str, max_hops: int = 3) -> list[dict]:
    """Real multi-hop: BFS from `start`, following entity->entity edges, until a
    `target_relation` edge is found. Returns each path that reaches the answer.

    Example: traverse(g, "widget", "SHIPS_FROM") walks
        widget --IS_A--> accessory --SHIPS_FROM--> Hanoi fulfillment center
    i.e. a 2-hop answer no single edge (and no single document chunk) contains.
    """
    from collections import deque

    results: list[dict] = []
    seen = {start}
    q: deque = deque([(start, [start])])
    while q:
        node, path = q.popleft()
        if len(path) > max_hops + 1:
            continue
        for rel, obj in graph.get(node, []):
            if rel == target_relation:
                results.append({"path": path + [obj], "hops": len(path), "answer": obj})
            if obj in graph and obj not in seen:   # obj is itself an entity -> keep hopping
                seen.add(obj)
                q.append((obj, path + [obj]))
    return results


def _sentence_chunks(docs_dir: Path) -> list[str]:
    """Sentence-level chunks (a common RAG granularity) across all docs."""
    chunks: list[str] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sent in re.split(r"(?<=[.!?])\s+", line):
                if sent.strip():
                    chunks.append(sent.strip())
    return chunks


def vector_foil(docs_dir: Path, subject: str, answer_hint: str) -> dict:
    """Show *why* flat chunk-retrieval RAG loses on a multi-hop question.

    A multi-hop fact is distributed across chunks: one chunk links the subject to
    a category, another links the category to the answer. No single chunk contains
    BOTH the subject and the answer, so top-1 retrieval can only ever return half
    the chain — regardless of embedding quality. The graph encodes the join, so a
    traversal returns the whole path. This contrast is the point of §13.
    """
    subject, answer_hint = subject.lower(), answer_hint.lower()
    chunks = _sentence_chunks(docs_dir)
    with_subject = [c for c in chunks if subject in c.lower()]
    with_answer = [c for c in chunks if answer_hint in c.lower()]
    complete = [c for c in chunks if subject in c.lower() and answer_hint in c.lower()]
    return {
        "n_chunks": len(chunks),
        "chunk_with_subject": with_subject[0] if with_subject else None,
        "chunk_with_answer": with_answer[0] if with_answer else None,
        "single_chunk_answers_it": bool(complete),   # False => flat RAG cannot bridge
    }
