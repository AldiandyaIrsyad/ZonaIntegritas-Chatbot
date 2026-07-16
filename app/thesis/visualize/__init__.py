"""Visualization tooling for the RAG ingestion and retrieval pipelines.

This package is **research tooling** (like :mod:`app.thesis._eval`), not
production thesis code. It intentionally imports from :mod:`app.kb.infra`
(real HTTP adapters) and :mod:`app.shared.db` (Base) to produce authentic
pipeline output for thesis presentation.

Usage::

    .venv/bin/python -m app.thesis.visualize.run \\
        --pdf-path datasets/permenpanrb-no-5-tahun-2024.pdf

Requires the real infrastructure services to be running:
    - Unstructured API  (port 8001)
    - Infinity         (port 7997)
    - Qdrant           (port 6333)
"""
