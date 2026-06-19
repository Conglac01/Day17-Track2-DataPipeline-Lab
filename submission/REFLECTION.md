# Reflection — Day 17 (≤ 200 words)

Answer briefly, in your own words. This is graded on reasoning, not length.

1. **The flywheel.** Day 13 emitted agent traces; today you turned them into an
   eval set and DPO pairs that Day 22 will train on. Which step in
   `traces → Bronze → datasets` would break most silently in production if you
   got it wrong — and how would you detect it?

2. **Decontamination.** Your run dropped 2 of 3 preference pairs because their
   prompts were in the eval set. What concretely goes wrong if you *skip* this
   step and train on those pairs? How would the lie show up in your metrics?

3. **Point-in-time.** The naive join leaked a future `lifetime_spend` into the
   training row. Describe one feature in a system you know that would be
   dangerous to join without an `ASOF`/point-in-time guard.

4. **Graph vs vector.** From `kg_demo.py`, name one question the knowledge graph
   answers well that flat chunk retrieval (`embed.py`) would struggle with, and
   one where the graph is overkill.

*Write your answers below.*

---

1. **The `split` tag assignment** would break most silently. If `split='eval'` is accidentally applied to all traces (or none), the eval set either balloons with training data or vanishes. Detection: add a CI check that asserts `len(eval_set)` is within expected bounds and that no trace_id appears in both eval and training sets. Monitor the eval/train overlap ratio — any non-zero value triggers an alert.

2. **The model memorizes the test.** Training on eval prompts means the model learns to regurgitate the reference answers verbatim. Eval metrics (accuracy, loss) look excellent because the model already "knows" those exact questions. In production, on novel prompts, performance drops sharply — the eval-prod gap widens. The lie is a silent inflation of all benchmark scores, discovered only when real users complain.

3. **"Number of support tickets in the last 30 days"** for a churn prediction model. If you join the *current* ticket count (including tickets filed *after* the churn event), you leak the consequence (customer complained, then churned) into the predictor. The model learns that high ticket count → churn, but at serving time you only have tickets filed *before* the prediction point. ASOF guards this by taking the ticket count known at-or-before each observation.

4. **Graph wins:** "Where does a widget ship from?" — requires joining widget→IS_A→accessory and accessory→SHIPS_FROM→Hanoi across two chunks. No single chunk holds the full chain. **Graph is overkill:** "What is the return policy for widgets?" — this fact lives in one sentence of one chunk. Flat vector retrieval finds it directly; traversing IS_A edges adds no value.
