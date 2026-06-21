# Thesis Evaluation Scripts

This directory contains standalone scripts for benchmarking and comparing
the thesis algorithms. All scripts are **pure** — they import from `app/thesis/`
and operate on CSV/JSON datasets, never touching the production database.

## Directory Structure

```
_eval/
├── README.md          ← This file
├── ivm/               ← Input Validation Module benchmarks
│   └── compare_strategies.py
├── rag/               ← Retrieval quality benchmarks
│   └── measure_retrieval.py
└── ram/               ← Response Assessment Module benchmarks
    └── compare_nli_models.py
```

## Running an Evaluation

```bash
# From repo root, with .venv activated
python -m app.thesis._eval.ivm.compare_strategies --dataset data/ivm_queries.csv
python -m app.thesis._eval.ram.compare_nli_models --dataset data/ram_pairs.csv
```

## Dataset Format

### IVM (`ivm_queries.csv`)
| query | label | top_k_scores |
|-------|-------|--------------|
| "..." | relevant/irrelevant | "[0.85, 0.72, ...]" |

### RAM (`ram_pairs.csv`)
| sentence | context | label |
|----------|---------|-------|
| "..." | "..." | entailment/neutral/contradiction |

### RAG (`rag_queries.csv`)
| query | relevant_doc_ids |
|-------|-----------------|
| "..." | "[uuid1, uuid2]" |
