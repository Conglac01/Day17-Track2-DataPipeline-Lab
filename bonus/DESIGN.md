# Bonus Challenge — Day 17: Feature Pipeline for Real-Time Fraud Detection on Vietnamese E-Commerce

## The Problem

A Vietnamese e-commerce platform (1.2M monthly active users, ~80K transactions/day) wants to deploy a real-time ML model that scores each checkout for fraud risk within **200ms P99**. The model consumes ~40 features — some real-time (session velocity, IP geolocation change), some batch-aggregated over days-to-weeks windows (user's 30-day dispute rate, merchant chargeback history), and some from third-party enrichment (device fingerprint, phone number risk DB). The engineering challenge is **not the model architecture** — it's the pipeline that feeds it features that are correct, fresh, and consistent between training and serving.

### Real Constraints

| Constraint | Detail |
|---|---|
| **Data volume** | ~80K transactions/day + 6M page-view events/day (clickstream). Peak: 3× average during Tết and 11.11 sales. |
| **Latency budget** | 200ms P99 for feature vector assembly at checkout. A timeout = declined transaction = lost revenue. |
| **Dirty inputs** | User-entered addresses contain 30%+ non-standard province names ("HCM" vs "TP. Hồ Chí Minh" vs "Sài Gòn"). Phone numbers mix 10 and 11 digits, with and without country code. |
| **Schema drift** | Payment gateway adds/removes fields in their callback JSON ~2×/year without notice. Garbage `status` values appear every few months. |
| **Regulation** | PDPL (Luật 91/2025) requires data localization for Vietnamese users' PII. User behavioral data must be deletable on request within 72h. |
| **Train/serve skew** | Features like "user's historical chargeback rate" must be point-in-time correct — a chargeback filed *after* today's transaction must not leak backward into training. |
| **Cost sensitivity** | Cloud infra budget of ~$3K/month. Real-time serving infra competes with batch training infra for the same budget. |

---

## Architecture Sketch

```
                          ┌────────────────────────┐
                          │     Feature Registry    │
                          │  (declarative YAML:     │
                          │   name, owner, window,  │
                          │   freshness_SLA, dtype) │
                          └────────────┬───────────┘
                                       │
   ┌──────────┐   ┌──────────┐   ┌────┴─────┐   ┌──────────────┐
   │Clickstream│   │Checkout  │   │3rd-party │   │Payment GW    │
   │(Kafka)    │   │API       │   │enrichment│   │callbacks     │
   └─────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘
         │              │              │                 │
         └──────────────┴──────────────┴─────────────────┘
                            │
                   ┌────────▼────────┐
                   │   Bronze Layer  │  append-only, raw JSON
                   │ (DuckDB + S3)   │  no transformation
                   └────────┬────────┘
                            │
              ┌─────────────┼──────────────┐
              │             │              │
     ┌────────▼────┐  ┌─────▼──────┐  ┌───▼──────────┐
     │ Streaming   │  │ Batch      │  │ Feature       │
     │ path        │  │ path       │  │ Validation    │
     │ (Flink)     │  │ (daily)    │  │ (Pandera)     │
     │             │  │            │  │               │
     │ Real-time   │  │ Window     │  │ schema checks │
     │ features:   │  │ features:  │  │ + distribution│
     │ velocity,   │  │ chargeback │  │ drift monitor  │
     │ geo-change  │  │ rate,      │  │               │
     │             │  │ dispute%   │  │ bad rows → DLQ│
     └──────┬──────┘  └─────┬──────┘  └───────┬───────┘
            │               │                 │
            └───────────────┼─────────────────┘
                            │
                   ┌────────▼────────┐
                   │  Feature Store  │
                   │  (online: Redis │
                   │   offline:      │
                   │   DuckDB/Parquet│
                   │   + ASOF-join)  │
                   └────────┬────────┘
                            │
              ┌─────────────┼──────────────┐
              │             │              │
     ┌────────▼────┐  ┌─────▼──────┐  ┌───▼──────────┐
     │ Online      │  │ Offline    │  │ Deletion     │
     │ Serving     │  │ Training   │  │ API          │
     │ (<200ms)    │  │ (ASOF PIT) │  │ (PDPL)       │
     └─────────────┘  └────────────┘  └──────────────┘
```

---

## Open Questions — Decisions & Tradeoffs

### Q1: Batch or Streaming? (Question #2)

**Decision: Lambda architecture — streaming fast path for 8 real-time features + daily batch for 32 aggregate features.**

**Tradeoff: Lambda vs Kappa.** A pure Kappa architecture (everything through Flink) would give us exactly-once semantics everywhere and simpler code — one codebase, one set of operators. However, computing a 90-day chargeback rate from raw events in a streaming join requires holding 90 days of state in Flink's RocksDB (estimated ~200GB for our scale), which would cost ~$1.2K/month in RAM alone. The daily batch path computes these long-window features on DuckDB in <3 minutes for ~$0.03/run and writes them to Redis with a simple `LAST_UPDATED` timestamp. The 8 real-time features (session velocity, IP geo-change, cart-value anomaly) have ≤5-min window state, keeping Flink state small (<2GB).

**Rejected alternative: pure batch (nightly feature refresh).** A fraud model running on 24h-stale features would miss the burst-attack pattern where a fraudster tests 50 stolen cards in 10 minutes. The platform's fraud loss data shows 40% of fraudulent transactions cluster within 30-minute windows. Stale features = blind model during the attack window.

### Q2: What Breaks at Scale? (Question #3)

**First bottleneck: small-file problem in the offline feature store.** Each daily batch run produces one Parquet file per feature per day. At 40 features × 365 days = 14,600 files/year. Point-in-time training queries that ASOF-join across all of them will hit S3 LIST latency. **Fix:** daily compaction — merge all feature files for a given day into a single `features_YYYY-MM-DD.parquet`, partitioned by `(user_id HASH % 64)`. Second bottleneck: Redis memory for the online store. With 1.2M active users × 40 features × ~200 bytes = ~9.6GB, fitting in a single `r7g.xlarge` (32GB). At 10× scale (12M users), we'd need Redis Cluster with sharding — budget impact ~$800 → ~$3K/month, triggering a cost review.

### Q3: Contracts & Quality (Question #4)

**Decision: Pandera schemas at every ingestion boundary + distribution drift monitor, with quarantine → dead-letter-queue → Slack alert.**

**Tradeoff: strict vs lazy validation.** Strict validation (fail on first bad row) would block the entire batch if 1 of 80K transactions has a weird `amount`. But silently dropping bad rows means missing fraud signals — a `status: "weird_gateway_edge_case"` might be the fraud pattern you needed to catch. **Chosen approach: lazy validation** (collect all failures) + **write bad rows to a queryable DLQ** so the fraud team can inspect them. A separate daily job counts DLQ records per `status` value and fires a Slack alert if any new/unseen `status` value appears or if the DLQ rate exceeds 0.5% of daily volume (a spike signals upstream schema drift).

### Q4: Train/Serve Parity (Question #5)

**Decision: All training features go through ASOF LEFT JOIN keyed on `(user_id, event_ts >= valid_from)`, NEVER a subquery that picks `MAX(valid_from)`.**

**Why this is load-bearing:** Consider a "user's 30-day chargeback rate" feature. On 2026-06-15, a transaction occurs. On 2026-06-20, that transaction is charged back. If training uses a naive `SELECT MAX(chargeback_rate)` without time-gating, the June-15 training row sees the June-20 chargeback — the model learns that transaction was risky *before the fraudster even disputed it*. In production on June-15, the chargeback hasn't happened yet, so the feature value is different → the model's prediction silently differs from what it learned. **This is the #1 silent regression in fraud ML.** Detection: compare ASOF vs naive feature distributions; any row where `feature_asof != feature_naive` is a leak. We assert `count(*) WHERE asof != naive == 0` in CI.

### Q5: Vietnamese Context (Question #10)

**Decision: Address normalization is a pipeline stage, not a model responsibility.**

Vietnamese addresses have canonical forms ("TP. Hồ Chí Minh", "Thành phố Hồ Chí Minh", "HCM", "Sài Gòn", "SG") but the fraud model needs a single `province_id`. We run a normalization stage that maps surface forms to a canonical province code (VN-ISO 3166-2) using a combination of deterministic rules (regex for common abbreviations) + a small fuzzy-matching model (TF-IDF + cosine on character n-grams, since Vietnamese syllables are ~1-3 characters). **This costs ~$15/month in compute vs the alternative** of letting the fraud model learn address embeddings end-to-end, which would require 50× more training data to learn a mapping a human already knows. PDPL also requires we track *which exact PII fields* feed which features for the deletion API — the normalization stage becomes the natural checkpoint for this lineage.

### Q6: Cost & Operations (Question #9)

**80% of the monthly cost is Redis (online feature store) + Flink cluster.** Redis: ~$500/month for a managed `r7g.xlarge`. Flink: ~$800/month for a 4-TaskManager cluster. Together they are $1,300 of a $3,000 budget. **Where to cut:** (1) Swap managed Flink for a self-hosted Flink on spot instances → ~$400/month savings but adds ~2h/month of DevOps toil. Acceptable because fraud pipeline downtime during non-peak hours is tolerable for <15min. (2) Swap managed Redis for a Valkey 8 self-hosted cluster → ~$250/month savings, but losing automatic failover. Rejected for now — the 200ms P99 SLA makes Redis availability load-bearing. If budget tightens, move long-tail features (used by <1% of transactions) from Redis to an in-process LRU cache backed by DuckDB.

---

## Rejected Architecture: Pure Feature-Store-as-a-Service (e.g., Tecton)

**Why rejected:** A managed feature store would handle point-in-time correctness, online/offline serving, and materialization automatically — eliminating most of the pipeline code above. However, at our scale (40 features, 1.2M users), Tecton's pricing (~$0.10 per 1,000 feature lookups × 80K transactions/day × 30 days ≈ $240/month for serving alone, plus materialization costs) would consume ~40% of our compute budget before we even pay for the actual feature computation. More critically, Tecton stores data in US/GCP regions — PDPL Article 26 requires Vietnamese-user behavioral data to remain in Vietnam (data localization). Running our own DuckDB+Redis stack on a Vietnamese-region VPS (e.g., Viettel Cloud, VNG Cloud) satisfies this requirement at lower cost.

---

## Minimal Prototype Extension

The `pipeline/features.py` module from this lab could be extended with a `normalize_vietnamese_address()` function that:

```python
def normalize_vietnamese_address(raw: str) -> str:
    """Map surface forms → canonical province code (VN-ISO 3166-2)."""
    # Stage 1: deterministic regex
    CANONICAL = {
        "hcm": "VN-SG", "sg": "VN-SG",
        "tp. hồ chí minh": "VN-SG", "thành phố hồ chí minh": "VN-SG",
        "sài gòn": "VN-SG", "tphcm": "VN-SG",
        "hn": "VN-HN", "hà nội": "VN-HN",
        # ... all 63 provinces
    }
    key = raw.strip().lower()
    return CANONICAL.get(key, key)  # fallthrough to fuzzy match

# Then wire it as a Bronze→Silver transform stage:
# raw_checkout → normalize_address → validate_schema → Silver
```

This illustrates the core decision (#5 above): normalize at ingestion, not at inference, so training and serving see identical address representations — one less source of train/serve skew in a pipeline where point-in-time leakage is already the #1 risk.
