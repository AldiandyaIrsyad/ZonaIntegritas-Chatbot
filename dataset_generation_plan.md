# Dataset Generation & Experiment Evaluation Plan

> **Purpose**: Evaluate the quality of the dataset generation pipeline, the generation plan for each subset, and whether the experiments actually address the research questions.

---

## Table of Contents

- [1. Dataset Quality Assessment](#1-dataset-quality-assessment)
  - [1.1 Overall Architecture Verdict](#11-overall-architecture-verdict)
  - [1.2 Issues Found & Fixes](#12-issues-found--fixes)
- [2. Subset Generation Plans](#2-subset-generation-plans)
  - [2.1 Subset A — RAG QA Triplets](#21-subset-a--rag-qa-triplets)
  - [2.2 Subset B — Adversarial Inputs](#22-subset-b--adversarial-inputs)
  - [2.3 Subset C — Boundary Relevance](#23-subset-c--boundary-relevance)
  - [2.4 Subset D — RAM Ground Truth](#24-subset-d--ram-ground-truth)
- [3. Experiment → Research Question Alignment](#3-experiment--research-question-alignment)
- [4. Summary of Actionable Fixes](#4-summary-of-actionable-fixes)

---

## 1. Dataset Quality Assessment

### 1.1 Overall Architecture Verdict

The Generator-Evaluator architecture is **well-designed** for a thesis-grade synthetic dataset:

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Generator-Evaluator separation | ✅ Good | DeepSeek V4 generates, 5-model panel validates — prevents self-reinforcing biases |
| Panel diversity | ✅ Good | 5 distinct model families (GLM, DeepSeek, Gemini, Llama, Mistral) reduce single-model blind spots |
| Fail-closed on error | ✅ Good | Panel errors default to NO; generator errors skip the batch |
| Blind injection / concordance | ✅ Good | 20% of 5/5-unanimous items are blind-injected for human verification ≥95% target |
| Temperature 0.0 | ✅ Good | Deterministic outputs for reproducibility |
| Threshold ≥4/5 | ✅ Good | Strict majority prevents borderline items from polluting the dataset |

**Overall: Solid foundation.** The issues below are fixable without changing the architecture.

### 1.2 Issues Found & Fixes

#### 🐛 ISSUE 1 — `exp2_retrieval/run.py` uses `RetrievalMetrics` with wrong field names (BUG)

**Severity**: 🔴 **Breaking** — this code will crash at runtime.

`exp2_retrieval/run.py` constructs `RetrievalMetrics` with fields `mrr` and `total`:
```python
# Line 119-125 — exp2_retrieval/run.py
return RetrievalMetrics(
    hit_rate_at_1=...,
    hit_rate_at_3=...,
    hit_rate_at_5=...,
    mrr=...,       # ← WRONG: field doesn't exist
    total=...,     # ← WRONG: field is called 'query_count'
)
```

But `metrics.py` defines `RetrievalMetrics` with `mrr_at_1`, `mrr_at_3`, `mrr_at_5`, and `query_count`:
```python
# metrics.py line 173-193
@dataclass(frozen=True)
class RetrievalMetrics:
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr_at_1: float      # ← NOT 'mrr'
    mrr_at_3: float
    mrr_at_5: float
    query_count: int      # ← NOT 'total'
```

Additionally, `compute_retrieval_metrics` is imported but **never used**; exp2 computes metrics manually.

**Fix**: Two options:
- **(Recommended)** Align `exp2_retrieval/run.py` to use `compute_retrieval_metrics` from metrics.py, or
- Simplify `RetrievalMetrics` to match what exp2 actually needs (single `mrr` field) and update the print report accordingly. The README only mentions MRR (not MRR@1/3/5), so a single `mrr` field is arguably more appropriate.

---

#### ⚠️ ISSUE 2 — `build_subset_a.py`: Document text fetch is fragile

**Severity**: 🟡 Medium

`build_subset_a.py` (Line 102-113) fetches document text by searching for `"zona integritas"` and filtering by `doc_id`:
```python
response = await client.get(
    "/api/kb/search",
    params={"q": "zona integritas", "top_k": 50},
)
results = response.json()
return "\n\n".join(
    r.get("text", "") for r in results if r.get("doc_id") == doc_id
)
```

**Problems**:
1. The hardcoded `"zona integritas"` query violates the domain-agnosticism principle from AGENTS.md — if the KB documents don't contain this exact term, the function returns empty text.
2. `top_k=50` may miss chunks if a document has more than 50 chunks.
3. This retrieves chunks, not full document text — the context may be incomplete.

**Fix**: Use a dedicated document content API (e.g., `/api/admin/pdfs/{id}/text`) if available, or change the search query to something more universal (e.g., use the document title as the search query). Since this is dataset generation for a *specific* thesis domain (ZI), the practical impact is low, but it's still worth improving.

---

#### ⚠️ ISSUE 3 — `build_subset_*.py` uses `print()` instead of `structlog`

**Severity**: 🟡 Low-Medium

All `build_subset_*.py` scripts use `print()` for progress output despite importing `structlog.get_logger()`. The `logger` variable is declared but unused.

**Fix**: Replace `print()` calls with `logger.info()` for progress and `logger.error()` for errors, or remove the unused `structlog` import.

---

#### ⚠️ ISSUE 4 — `exp4_end_to_end/run.py`: No-guardrail baseline is a no-op

**Severity**: 🟡 Medium

`run_no_guardrail_pipeline` (Line 241-263) simply calls `run_pipeline()` with no changes:
```python
async def run_no_guardrail_pipeline(...) -> PipelineResult:
    # The baseline uses the same endpoint...
    return await run_pipeline(api_url, row, session_id)
```

This means the "with guardrails" and "no guardrails" results will be **identical**, rendering the experiment useless for comparison.

**Fix**: Either:
1. Add a `?skip_guardrails=true` query param to the chat API that disables IVM + RAM, or
2. Run a second app instance with `IVM_ENABLED=false` and `RAM_ENABLED=false` environment variables, or
3. Call the LLM directly (bypassing the chat endpoint) for the baseline, sending the same retrieved context but skipping IVM validation and RAM assessment.

This is a **known TODO** (the comment in the code acknowledges it), but it must be resolved before the experiment can produce meaningful results.

---

#### ⚠️ ISSUE 5 — `exp4_end_to_end/run.py`: API request body mismatch

**Severity**: 🟡 Medium

exp4 (Line 183-187) sends `{"content": row.question}` to `/api/chat/sessions/{id}/messages`:
```python
resp = await client.post(
    f"/api/chat/sessions/{session_id}/messages",
    json={"content": row.question},
)
```

But `build_subset_d.py` (Line 106-109) sends `{"message": question}` to `/api/chat/sessions/{id}/stream`:
```python
response = await client.post(
    f"/api/chat/sessions/{session_id}/stream",
    json={"message": question},
)
```

These use **different endpoints** (`/messages` vs `/stream`) and **different request schemas** (`content` vs `message`). One of them is wrong, or the API supports both. This needs verification against the actual chat API.

**Fix**: Check `app/chat/api.py` for the correct endpoint and request schema, then unify both scripts.

---

#### ⚠️ ISSUE 6 — `exp4_end_to_end/run.py`: Looks for event type `"token"` but Subset D uses `"chunk"`

**Severity**: 🟡 Medium

exp4 (Line 207) checks for `chunk.get("type") == "token"`, while `build_subset_d.py` (Line 124) checks for `chunk_type == "chunk"`. One is correct; the other will silently produce empty responses.

**Fix**: Verify the actual NDJSON event type emitted by the streaming endpoint and unify.

---

#### 💡 ISSUE 7 — Subset A `out-of-domain` questions: `source_doc_id` set to `"NONE"` breaks retrieval evaluation

**Severity**: 🟡 Medium

In `build_subset_a.py` (Line 255), out-of-domain questions get `source_doc_id = "NONE"`. But in `exp2_retrieval/run.py` (Line 99), the evaluation checks `relevant_id = row.source_doc_id` — so out-of-domain rows will always produce a miss (Hit Rate=0, MRR=0), dragging down the overall metrics without meaningful insight.

**Fix**: Filter out `out-of-domain` rows from Subset A when running Experiment 2 (retrieval evaluation). This is a one-line change:
```python
dataset = [r for r in dataset if r.category != "out-of-domain"]
```

---

#### 💡 ISSUE 8 — Dataset sizes are small for statistical significance

**Severity**: 🟢 Low (acceptable for thesis)

| Subset | Default Count | Items per Category |
|--------|--------------|-------------------|
| A | 85 (default), 100 (CLI) | factual: 30, procedural: 25, multi-hop: 20, OOD: 10 |
| B | 100 | jailbreak: 15, DAN: 15, hidden: 20, safe_normal: 25, safe_complex: 25 |
| C | 60 | direct_zi: 15, indirect_zi: 15, near_miss: 10, adjacent: 10, off_topic: 10 |
| D | 30 questions | Sentence count depends on response length |

These sizes are acceptable for a thesis pilot study but marginal for statistical significance. The bootstrap CIs will be wide. This is noted, not a blocking issue.

---

## 2. Subset Generation Plans

### 2.1 Subset A — RAG QA Triplets

| Aspect | Plan | Assessment |
|--------|------|------------|
| **Source** | KB documents via `/api/admin/pdfs` + `/api/kb/search` | ⚠️ Search-based text fetch is fragile (Issue 2) |
| **Categories** | factual, procedural, multi-hop, out-of-domain | ✅ Good coverage of question types |
| **Generation** | Per-document, per-category batches of 5 | ✅ Small batches improve diversity |
| **Validation** | Panel evaluates: answerable? hallucination-free? | ✅ Appropriate validation criteria |
| **Balance** | Category targets (30/25/20/10), round-robin across documents | ✅ Good balance strategy |
| **Output schema** | `question, category, ground_truth_answer, source_doc_id, source_context` | ✅ Complete for downstream experiments |
| **Used by** | Exp 2 (retrieval), Exp 4 (end-to-end) | ✅ |

**Verdict**: Solid plan. Fix Issue 2 (fragile text fetch) and Issue 7 (OOD in retrieval eval).

### 2.2 Subset B — Adversarial Inputs

| Aspect | Plan | Assessment |
|--------|------|------------|
| **Source** | Synthetic (no KB needed) | ✅ Correct — adversarial inputs don't come from KB |
| **Attack types** | jailbreak (15), DAN (15), hidden_instruction (20), safe_normal (25), safe_complex (25) | ✅ Good attack taxonomy |
| **Generation** | Per-attack-type batches | ✅ Clear separation |
| **Validation** | Panel reclassifies safe/malicious — accepted if ≥4/5 agree | ✅ Strong validation |
| **Balance** | 50 malicious / 50 safe (balanced) | ✅ |
| **Output schema** | `query, label, attack_type` | ✅ Matches Tabel 3.4 |
| **Used by** | Exp 1a (safety) | ✅ |

**Verdict**: ✅ No issues found. Clean and well-structured.

### 2.3 Subset C — Boundary Relevance

| Aspect | Plan | Assessment |
|--------|------|------------|
| **Source** | Synthetic (no KB needed) | ✅ |
| **Subtypes** | direct_zi (15), indirect_zi (15), near_miss_govt (10), adjacent_legal (10), off_topic (10) | ✅ Good boundary taxonomy |
| **Generation** | Per-subtype batches | ✅ |
| **Validation** | Panel labels in_domain/out_of_domain — accepted if ≥4/5 agree | ✅ |
| **Balance** | 30 in-domain / 30 out-of-domain | ✅ |
| **Output schema** | `query, label, subtype` | ✅ Matches Tabel 3.5 |
| **Used by** | Exp 1b (relevance) | ✅ |

**Verdict**: ✅ No issues found. Well-balanced between easy and hard cases.

### 2.4 Subset D — RAM Ground Truth

| Aspect | Plan | Assessment |
|--------|------|------------|
| **Source** | Derived from Subset A — runs questions through the live pipeline | ✅ Realistic responses for NLI evaluation |
| **Sentence splitting** | Regex: `(?<=[.!?])\s+` | ⚠️ May mis-split on abbreviations (e.g., "Permen No. 12") or numbered lists, but acceptable |
| **Label taxonomy** | supported, partially_supported, not_supported, no_source_needed | ✅ 4-label → 3-class NLI mapping is well-defined |
| **Validation** | Panel assigns labels (evaluate_label) — accepted if ≥4/5 agree on same label | ✅ Strong inter-annotator agreement requirement |
| **Dependency** | Requires: (a) Subset A already generated, (b) live app running with KB ingested | ⚠️ Complex dependency chain |
| **Output schema** | 8 columns including sentence-level annotation | ✅ Matches Tabel 3.6 |
| **Used by** | Exp 3 (RAM) | ✅ |

**Verdict**: Good design. The dependency on a running pipeline makes this the most complex subset to generate. The sentence splitting regex is good enough for Indonesian formal text.

---

## 3. Experiment → Research Question Alignment

### Research Questions (from RQ.md)

| RQ | Indonesian | English Translation |
|----|-----------|---------------------|
| **RQ1** | Bagaimana merancang chatbot RAG dengan guardrails ganda untuk diseminasi pedoman Zona Integritas di UPI? | How to design a RAG chatbot with dual guardrails for ZI guideline dissemination at UPI? |
| **RQ2** | Bagaimana efektivitas Input Validation Module (IVM) dalam mendeteksi dan menolak input berbahaya serta kueri di luar domain? | How effective is the IVM in detecting and rejecting dangerous inputs and out-of-domain queries? |
| **RQ3** | Bagaimana efektivitas Response Assessment Module (RAM) berbasis NLI dalam memverifikasi kesesuaian respons terhadap dokumen sumber? | How effective is the NLI-based RAM in verifying response conformity to source documents? |
| **RQ4** | Bagaimana performa keseluruhan chatbot RAG berguardrail ganda dalam menjawab pertanyaan tentang pedoman Zona Integritas? | How is the overall performance of the dual-guardrail RAG chatbot in answering ZI questions? |

### Alignment Matrix

| Experiment | Dataset | What It Measures | RQ Addressed | Alignment Quality |
|------------|---------|-----------------|--------------|-------------------|
| **Exp 1a** (Safety) | Subset B | SLM vs prompting baseline for malicious input detection | **RQ2** (partially) | ✅ Directly tests IVM safety gate — the "mendeteksi input berbahaya" part of RQ2 |
| **Exp 1b** (Relevance) | Subset C | LLM-as-Judge vs keyword overlap for domain relevance | **RQ2** (partially) | ✅ Directly tests IVM relevance gate — the "menolak kueri di luar domain" part of RQ2 |
| **Exp 2** (Retrieval) | Subset A | Hybrid vs dense vs sparse retrieval quality | **RQ4** (indirectly) | 🟡 Supports RQ4 by validating retrieval quality, but **no RQ explicitly asks about retrieval**. See note below. |
| **Exp 3** (RAM) | Subset D | NLI vs token-Jaccard for hallucination detection | **RQ3** | ✅ Directly tests RAM effectiveness — exactly what RQ3 asks |
| **Exp 4** (End-to-End) | Subset A | Full pipeline vs no-guardrail baseline | **RQ4**, **RQ1** | ⚠️ Intended to answer RQ4, but the no-guardrail baseline is currently a no-op (Issue 4). Also implicitly addresses RQ1 by demonstrating the full system design. |

### Detailed RQ Coverage Analysis

#### RQ1: "How to design the chatbot?"

RQ1 is a **design question**, not an empirical question. It is answered by the system architecture description (§3), not by any single experiment. No experiment directly answers "how to design" — this is normal for a design-oriented RQ.

**Assessment**: ✅ RQ1 is adequately covered by the architecture chapters. Experiment 4 indirectly validates the design by showing end-to-end performance.

#### RQ2: "How effective is the IVM?"

**Coverage**: Experiments 1a + 1b together **fully cover** RQ2.
- Exp 1a covers the safety gate (malicious input detection)
- Exp 1b covers the relevance gate (out-of-domain rejection)

**Assessment**: ✅ Excellent coverage. The two sub-experiments decompose the two IVM functions cleanly.

#### RQ3: "How effective is the NLI-based RAM?"

**Coverage**: Experiment 3 **directly answers** RQ3.

**Assessment**: ✅ Direct and complete. The NLI model is compared against a token-Jaccard baseline, and results are reported per-class with Cohen's Kappa for agreement quality.

#### RQ4: "How is the overall chatbot performance?"

**Coverage**: Experiment 4 is intended to answer this, **but has critical issues**:

- **Issue**: The no-guardrail baseline in exp4 is a no-op (Issue 4). Without a real baseline, you cannot claim "the guardrails improved performance." The experiment would only show absolute performance, not comparative advantage.
- **Gap**: Experiment 2 (retrieval) tests retrieval quality but is not explicitly tied to any RQ. If retrieval is part of the "overall performance" story, it should be discussed under RQ4. Otherwise, RQ4 only has BERTScore F1, Faithfulness, and Abstention Accuracy — which is acceptable but would be strengthened by incorporating retrieval results.

**Assessment**: 🟡 Partially covered. Needs the no-guardrail baseline to be fixed for meaningful comparison.

### Missing or Weak Links

| Gap | Impact | Suggestion |
|-----|--------|------------|
| Exp 2 (retrieval) is not tied to any specific RQ | The experiment exists but has no explicit RQ anchor | Either: (a) add retrieval quality as part of RQ4's "overall performance" discussion, or (b) add a sub-question to RQ1 about retrieval strategy selection |
| Exp 4 baseline is a no-op | RQ4 cannot be meaningfully answered without a comparison | Must fix before running the experiment (Issue 4) |
| No experiment tests the **combined** IVM (safety + relevance together) | RQ2 asks about IVM effectiveness holistically, but exp 1a and 1b test components separately | Consider adding a "combined IVM" row in the final RQ2 discussion that shows what happens when both gates are active |

---

## 4. Summary of Actionable Fixes

### Priority 1: Must Fix Before Running Experiments

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | `RetrievalMetrics` field name mismatch (runtime crash) | `exp2_retrieval/run.py` | Align field names with `metrics.py` or simplify `RetrievalMetrics` |
| 4 | No-guardrail baseline is a no-op | `exp4_end_to_end/run.py` | Implement a real bypass (skip IVM+RAM via API flag or separate instance) |
| 5 | API request body mismatch (`content` vs `message`) | `exp4_end_to_end/run.py` | Verify correct endpoint/schema from `chat/api.py` |
| 6 | NDJSON event type mismatch (`token` vs `chunk`) | `exp4_end_to_end/run.py` | Verify and unify event type |

### Priority 2: Should Fix for Quality

| # | Issue | File | Fix |
|---|-------|------|-----|
| 2 | Fragile document text fetch (hardcoded query) | `build_subset_a.py` | Use document title or dedicated API |
| 7 | OOD rows pollute retrieval metrics | `exp2_retrieval/run.py` | Filter out `out-of-domain` rows before retrieval eval |

### Priority 3: Nice to Have

| # | Issue | File | Fix |
|---|-------|------|-----|
| 3 | `print()` vs `structlog` inconsistency | `build_subset_*.py` | Use structlog for events, keep print for interactive progress |
| 8 | Small dataset sizes | All subsets | Document limitations, note wide CIs |
| — | Exp 2 not anchored to an explicit RQ | Writing/analysis | Discuss retrieval under RQ4 in the thesis body |
