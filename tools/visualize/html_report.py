"""HTML report builder for pipeline visualization.

Generates a self-contained HTML file (inline CSS, no external dependencies)
from a :class:`PipelineSnapshot`. The report shows every stage of the
ingestion and retrieval pipelines with color-coded content-type badges,
collapsible sections, and formatted text samples.

Designed for thesis presentation: "this is the output example of document A".
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import List

from .capture import (
    ChildChunkSnapshot,
    EmbeddingSnapshot,
    IngestionSnapshot,
    ParentChunkSnapshot,
    ParsedElementSnapshot,
    PipelineSnapshot,
    QdrantPointSnapshot,
    RetrievedParentSnapshot,
    RetrievalSnapshot,
    SearchResultSnapshot,
)

# ── Color mapping for content types ──────────────────────────────────
_CONTENT_TYPE_COLORS: dict[str, str] = {
    "text": "#3b82f6",      # blue
    "table": "#22c55e",     # green
    "figure": "#f97316",    # orange
    "hybrid": "#a855f7",    # purple
}

# Max text length to display inline before truncating with a toggle
_MAX_INLINE_TEXT = 2000


def build_html(snapshot: PipelineSnapshot) -> str:
    """Build a complete self-contained HTML document from a snapshot.

    Args:
        snapshot: The combined ingestion + retrieval snapshot.

    Returns:
        A full HTML string (with inline CSS, no external dependencies).
    """
    parts: List[str] = []
    parts.append(_HTML_HEAD)
    parts.append(_build_header(snapshot))
    parts.append(_build_ingestion_section(snapshot.ingestion))
    parts.append(_build_retrieval_section(snapshot.retrieval))
    parts.append(_HTML_FOOTER)
    return "\n".join(parts)


def write_html(snapshot: PipelineSnapshot, output_path: str) -> str:
    """Build the HTML report and write it to a file.

    Args:
        snapshot: The combined pipeline snapshot.
        output_path: Absolute path to write the HTML file.

    Returns:
        The path that was written to.
    """
    html_content = build_html(snapshot)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_content, encoding="utf-8")
    return output_path


# ── HTML fragments ───────────────────────────────────────────────────

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Pipeline Visualization</title>
<style>
  :root {
    --bg: #0f172a;
    --card-bg: #1e293b;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #334155;
    --accent: #38bdf8;
    --code-bg: #0f172a;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
    line-height: 1.6;
  }
  h1 { color: var(--accent); border-bottom: 2px solid var(--border); padding-bottom: 8px; }
  h2 { color: var(--accent); margin-top: 32px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  h3 { color: #cbd5e1; margin-top: 24px; }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
  }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: 600;
    color: #fff;
    text-transform: uppercase;
  }
  .badge-text { background: #3b82f6; }
  .badge-table { background: #22c55e; }
  .badge-figure { background: #f97316; }
  .badge-hybrid { background: #a855f7; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.875em;
  }
  th, td {
    text-align: left;
    padding: 8px 12px;
    border: 1px solid var(--border);
  }
  th { background: #334155; color: #f1f5f9; }
  tr:nth-child(even) { background: #1e293b; }
  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.8125em;
    max-height: 400px;
    overflow-y: auto;
  }
  details {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin: 8px 0;
    padding: 8px 12px;
  }
  summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--accent);
    padding: 4px 0;
  }
  details[open] summary { margin-bottom: 8px; }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    margin: 16px 0;
  }
  .stat-box {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
  }
  .stat-value { font-size: 1.75em; font-weight: 700; color: var(--accent); }
  .stat-label { font-size: 0.75em; color: var(--text-muted); text-transform: uppercase; }
  .breadcrumbs {
    color: var(--text-muted);
    font-size: 0.8125em;
    margin: 4px 0;
  }
  .score-bar {
    display: inline-block;
    height: 8px;
    background: var(--accent);
    border-radius: 4px;
    vertical-align: middle;
    margin-right: 6px;
  }
  .mono { font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 0.8125em; }
  .muted { color: var(--text-muted); }
  .truncate { max-height: 200px; overflow-y: auto; }
</style>
</head>
<body>
"""

_HTML_FOOTER = """
</body>
</html>
"""


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(text)


def _badge(content_type: str) -> str:
    """Render a content-type badge."""
    color = _CONTENT_TYPE_COLORS.get(content_type, "#64748b")
    return f'<span class="badge" style="background:{color}">{_esc(content_type)}</span>'


def _breadcrumbs(crumbs: List[str]) -> str:
    """Render breadcrumbs as a muted path string."""
    if not crumbs:
        return ""
    return f'<div class="breadcrumbs">📍 {" &gt; ".join(_esc(c) for c in crumbs)}</div>'


def _build_header(snapshot: PipelineSnapshot) -> str:
    """Build the report header with summary stats."""
    ing = snapshot.ingestion
    ret = snapshot.retrieval
    return f"""
<h1>🔍 RAG Pipeline Visualization</h1>
<p class="muted">Generated: {_esc(snapshot.timestamp)}</p>

<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-value">{len(ing.elements)}</div>
    <div class="stat-label">Parsed Elements</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{len(ing.parents)}</div>
    <div class="stat-label">Parent Chunks</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{len(ing.children)}</div>
    <div class="stat-label">Child Chunks</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{ing.qdrant_point_count}</div>
    <div class="stat-label">Qdrant Vectors</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{len(ret.hybrid_results)}</div>
    <div class="stat-label">Retrieved (Hybrid)</div>
  </div>
</div>
"""


def _build_ingestion_section(ing: IngestionSnapshot) -> str:
    """Build the full ingestion pipeline section."""
    parts: List[str] = []
    parts.append("<h2>📥 Ingestion Pipeline</h2>")

    # Document metadata
    parts.append(f"""
<div class="card">
  <h3>📄 Document Metadata</h3>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Doc ID</td><td class="mono">{_esc(ing.doc_id)}</td></tr>
    <tr><td>Title</td><td>{_esc(ing.doc_title)}</td></tr>
    <tr><td>PDF Path</td><td class="mono">{_esc(ing.pdf_path)}</td></tr>
    <tr><td>SQLite DB</td><td class="mono">{_esc(ing.sqlite_path)}</td></tr>
    <tr><td>Qdrant Collection</td><td class="mono">{_esc(ing.qdrant_collection)}</td></tr>
    <tr><td>Total Element Chars</td><td>{ing.total_element_chars:,}</td></tr>
    <tr><td>Total Parent Chars</td><td>{ing.total_parent_chars:,}</td></tr>
    <tr><td>Total Child Chars</td><td>{ing.total_child_chars:,}</td></tr>
  </table>
</div>
""")

    # Stage 1: Parsed Elements
    parts.append(_build_parsed_elements(ing))

    # Stage 2: Parent Chunks
    parts.append(_build_parent_chunks(ing))

    # Stage 3: SQLite Database
    parts.append(_build_sqlite_section(ing))

    # Stage 4: Child Chunks
    parts.append(_build_child_chunks(ing))

    # Stage 5: Embeddings
    parts.append(_build_embeddings(ing))

    # Stage 6: Qdrant Vectors
    parts.append(_build_qdrant_vectors(ing))

    return "\n".join(parts)

def _build_parsed_elements(ing: IngestionSnapshot) -> str:
    """Build the parsed elements table."""
    rows: List[str] = []
    for el in ing.elements:
        rows.append(f"""<tr>
  <td>{el.index}</td>
  <td><strong>{_esc(el.element_type)}</strong></td>
  <td>{el.text_length:,}</td>
  <td>{el.page if el.page is not None else "—"}</td>
  <td class="mono">{_esc(el.text_preview)}</td>
  <td class="muted">{_esc(", ".join(el.metadata_keys))}</td>
</tr>""")

    # Element type distribution
    dist_bars = []
    max_count = max(ing.element_type_counts.values()) if ing.element_type_counts else 1
    for etype, count in sorted(
        ing.element_type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        bar_width = int((count / max_count) * 100)
        dist_bars.append(
            f"""<div style="margin:4px 0">
  <span style="display:inline-block;width:120px">{_esc(etype)}</span>
  <span class="score-bar" style="width:{bar_width}px"></span>
  <strong>{count}</strong>
</div>"""
        )

    return f"""
<div class="card">
  <h3>Stage 1 — Parsed Elements (Unstructured API)</h3>
  <p class="muted">Total: {len(ing.elements)} elements</p>
  <div>{"".join(dist_bars)}</div>
  <details>
    <summary>📋 Full Element Table ({len(ing.elements)} rows)</summary>
    <table>
      <tr><th>#</th><th>Type</th><th>Chars</th><th>Page</th><th>Text Preview</th><th>Metadata Keys</th></tr>
      {"".join(rows)}
    </table>
  </details>
</div>
"""


def _build_parent_chunks(ing: IngestionSnapshot) -> str:
    """Build the parent chunks section with collapsible cards."""
    cards: List[str] = []
    for pc in ing.parents:
        text_display = _esc(pc.text)
        if len(pc.text) > _MAX_INLINE_TEXT:
            text_display = (
                f'<details><summary>Show full text ({pc.text_length:,} chars)</summary>'
                f'<pre>{text_display}</pre></details>'
            )
            preview = _esc(pc.text[:500]) + "…"
        else:
            preview = text_display

        cards.append(f"""
<div class="card">
  <div>
    <strong>Parent #{pc.chunk_index}</strong>
    {_badge(pc.content_type)}
    <span class="muted mono">id: {pc.id[:12]}…</span>
    <span class="muted">| Page {pc.page if pc.page is not None else "—"}</span>
    <span class="muted">| {pc.text_length:,} chars</span>
    <span class="muted">| Depth: {pc.depth}</span>
    <span class="muted">| Ordinal: {pc.ordinal}</span>
  </div>
  {_breadcrumbs(pc.breadcrumbs)}
  {f'<div class="muted mono">📍 Path: {_esc(pc.path)}</div>' if pc.path else ''}
  {f'<div class="muted mono">⬆ Parent: {_esc(pc.parent_id[:12])}…</div>' if pc.parent_id else ''}
  <pre class="truncate">{preview}</pre>
  {text_display if len(pc.text) > _MAX_INLINE_TEXT else ''}
</div>""")

    # Content type distribution
    ct_bars = []
    max_ct = max(ing.content_type_counts.values()) if ing.content_type_counts else 1
    for ct, count in sorted(
        ing.content_type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        bar_width = int((count / max_ct) * 100)
        ct_bars.append(
            f"""<div style="margin:4px 0">
  {_badge(ct)}
  <span class="score-bar" style="width:{bar_width}px"></span>
  <strong>{count}</strong>
</div>"""
        )

    return f"""
<div class="card">
  <h3>Stage 2 — Parent Chunks (Hierarchical Chunking)</h3>
  <p class="muted">Total: {len(ing.parents)} parent chunks</p>
  <div>{"".join(ct_bars)}</div>
  {"".join(cards)}
</div>
"""


def _build_sqlite_section(ing: IngestionSnapshot) -> str:
    """Build the SQLite database rows section."""
    rows: List[str] = []
    for pc in ing.parents:
        rows.append(f"""<tr>
  <td class="mono">{_esc(pc.id[:12])}…</td>
  <td class="mono">{_esc(ing.doc_id[:12])}…</td>
  <td>{pc.chunk_index}</td>
  <td>{pc.page if pc.page is not None else "—"}</td>
  <td>{_badge(pc.content_type)}</td>
  <td class="muted">{_esc(" > ".join(pc.breadcrumbs)) if pc.breadcrumbs else "—"}</td>
  <td>{pc.text_length:,}</td>
</tr>""")

    return f"""
<div class="card">
  <h3>Stage 3 — Database (SQLite: parent_chunks table)</h3>
  <p class="muted">
    📁 SQLite file: <code>{_esc(ing.sqlite_path)}</code><br>
    Open with <strong>DB Browser for SQLite</strong> to inspect all tables:
    <code>pdf_documents</code>, <code>parent_chunks</code>, <code>ingestion_tasks</code>.
  </p>
  <table>
    <tr>
      <th>ID</th><th>Doc ID</th><th>Index</th><th>Page</th>
      <th>Type</th><th>Breadcrumbs</th><th>Chars</th>
    </tr>
    {"".join(rows)}
  </table>
</div>
"""


def _build_child_chunks(ing: IngestionSnapshot) -> str:
    """Build the child chunks section, grouped by parent."""
    # Group children by parent_index
    by_parent: dict[int, List[ChildChunkSnapshot]] = {}
    for child in ing.children:
        by_parent.setdefault(child.parent_index, []).append(child)

    groups: List[str] = []
    for parent_idx in sorted(by_parent.keys()):
        children = by_parent[parent_idx]
        child_rows: List[str] = []
        for c in children:
            child_rows.append(f"""<tr>
  <td class="mono">{_esc(c.id[:12])}…</td>
  <td>{c.text_length}</td>
  <td>{_badge(c.content_type)}</td>
  <td><pre class="truncate" style="max-height:100px">{_esc(c.text)}</pre></td>
</tr>""")
        groups.append(f"""
<details>
  <summary>Parent #{parent_idx} → {len(children)} child chunk(s)</summary>
  <table>
    <tr><th>Child ID</th><th>Chars</th><th>Type</th><th>Text (with context prefix)</th></tr>
    {"".join(child_rows)}
  </table>
</details>""")

    return f"""
<div class="card">
  <h3>Stage 4 — Child Chunks (Sentence-level Splitting)</h3>
  <p class="muted">
    Total: {len(ing.children)} child chunks across {len(by_parent)} parents.<br>
    Each child includes a leading breadcrumb tag (e.g. <code>BAB II &gt; Pasal 5</code>) for independent embedding.
  </p>
  {"".join(groups)}
</div>
"""


def _build_embeddings(ing: IngestionSnapshot) -> str:
    """Build the embeddings table."""
    rows: List[str] = []
    for emb in ing.embeddings[:50]:  # Cap at 50 for readability
        dense_str = ", ".join(f"{v:.4f}" for v in emb.dense_preview)
        sparse_str = ", ".join(
            f"({idx}: {val:.4f})" for idx, val in emb.sparse_top5
        )
        rows.append(f"""<tr>
  <td class="mono">{_esc(emb.child_id[:12])}…</td>
  <td>{emb.dense_dim}</td>
  <td class="mono">[{dense_str}, …]</td>
  <td>{emb.sparse_nnz}</td>
  <td class="mono">{sparse_str}</td>
</tr>""")

    remaining = len(ing.embeddings) - 50
    note = f'<p class="muted">Showing first 50 of {len(ing.embeddings)} embeddings.</p>' if remaining > 0 else ''

    return f"""
<div class="card">
  <h3>Stage 5 — Embeddings (BGE-M3: Dense + Sparse BM25)</h3>
  <p class="muted">
    Each child chunk is embedded into a {ing.embeddings[0].dense_dim if ing.embeddings else 1024}-dim dense vector
    and a sparse BM25 vector (token ID → weight).
  </p>
  {note}
  <table>
    <tr>
      <th>Child ID</th><th>Dense Dim</th><th>Dense Preview (first 8)</th>
      <th>Sparse NNZ</th><th>Top-5 Sparse Tokens</th>
    </tr>
    {"".join(rows)}
  </table>
</div>
"""


def _build_qdrant_vectors(ing: IngestionSnapshot) -> str:
    """Build the Qdrant vectors section."""
    rows: List[str] = []
    for qp in ing.qdrant_points[:50]:
        rows.append(f"""<tr>
  <td class="mono">{_esc(qp.point_id[:12])}…</td>
  <td class="mono">{_esc(qp.parent_chunk_id[:12])}…</td>
  <td class="mono">{_esc(qp.doc_id[:12])}…</td>
  <td>{_badge(qp.content_type)}</td>
  <td>{qp.dense_dim}</td>
  <td>{qp.sparse_nnz}</td>
</tr>""")

    remaining = len(ing.qdrant_points) - 50
    note = f'<p class="muted">Showing first 50 of {len(ing.qdrant_points)} points.</p>' if remaining > 0 else ''

    return f"""
<div class="card">
  <h3>Stage 6 — Qdrant Vectors (Hybrid Index)</h3>
  <p class="muted">
    Collection: <code>{_esc(ing.qdrant_collection)}</code><br>
    Total points: <strong>{ing.qdrant_point_count}</strong><br>
    Each point stores a named dense vector ("dense", 1024-dim cosine) and a
    sparse vector ("bm25", with IDF modifier). Payload includes
    <code>parent_chunk_id</code>, <code>doc_id</code>, <code>is_active</code>,
    <code>breadcrumbs</code>, <code>content_type</code>.
  </p>
  {note}
  <table>
    <tr>
      <th>Point ID</th><th>Parent Chunk ID</th><th>Doc ID</th>
      <th>Type</th><th>Dense Dim</th><th>Sparse NNZ</th>
    </tr>
    {"".join(rows)}
  </table>
</div>
"""


def _build_retrieval_section(ret: RetrievalSnapshot) -> str:
    """Build the retrieval pipeline section."""
    parts: List[str] = []
    parts.append("<h2>🔎 Retrieval Pipeline</h2>")

    # Query + embedding
    dense_preview = ", ".join(f"{v:.4f}" for v in ret.query_dense_preview)
    parts.append(f"""
<div class="card">
  <h3>Query & Embedding</h3>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Query</td><td><strong>{_esc(ret.query)}</strong></td></tr>
    <tr><td>Dense Preview</td><td class="mono">[{dense_preview}, …]</td></tr>
    <tr><td>Sparse NNZ</td><td>{ret.query_sparse_nnz}</td></tr>
  </table>
</div>
""")

    # 3-mode comparison
    parts.append(_build_mode_comparison(ret))

    # Retrieved parent chunks
    parts.append(_build_retrieved_parents(ret))

    return "\n".join(parts)


def _build_mode_comparison(ret: RetrievalSnapshot) -> str:
    """Build the 3-mode search comparison table."""
    # Build a unified table: chunk_id | dense_rank | dense_score | sparse_rank | sparse_score | hybrid_rank | rrf_score
    all_chunk_ids: List[str] = []
    seen = set()
    for mode_results in [ret.dense_results, ret.sparse_results, ret.hybrid_results]:
        for sr in mode_results:
            if sr.chunk_id not in seen:
                all_chunk_ids.append(sr.chunk_id)
                seen.add(sr.chunk_id)

    dense_map = {sr.chunk_id: sr for sr in ret.dense_results}
    sparse_map = {sr.chunk_id: sr for sr in ret.sparse_results}
    hybrid_map = {sr.chunk_id: sr for sr in ret.hybrid_results}

    rows: List[str] = []
    for cid in all_chunk_ids:
        d = dense_map.get(cid)
        s = sparse_map.get(cid)
        h = hybrid_map.get(cid)
        rows.append(f"""<tr>
  <td class="mono">{_esc(cid[:12])}…</td>
  <td>{f"#{d.rank}" if d else "—"}</td>
  <td>{f"{d.score:.4f}" if d else "—"}</td>
  <td>{f"#{s.rank}" if s else "—"}</td>
  <td>{f"{s.score:.4f}" if s else "—"}</td>
  <td>{f"#{h.rank}" if h else "—"}</td>
  <td><strong>{f"{h.score:.4f}" if h else "—"}</strong></td>
</tr>""")

    return f"""
<div class="card">
  <h3>Hybrid Search — 3-Mode Comparison (RRF Fusion)</h3>
  <p class="muted">
    The same query is searched in three modes to illustrate Reciprocal Rank Fusion (RRF).
    Dense = cosine similarity (BGE-M3), Sparse = BM25, Hybrid = RRF-fused.
  </p>
  <table>
    <tr>
      <th rowspan="2">Chunk ID</th>
      <th colspan="2" style="text-align:center">Dense</th>
      <th colspan="2" style="text-align:center">Sparse (BM25)</th>
      <th colspan="2" style="text-align:center">Hybrid (RRF)</th>
    </tr>
    <tr>
      <th>Rank</th><th>Score</th>
      <th>Rank</th><th>Score</th>
      <th>Rank</th><th>Score</th>
    </tr>
    {"".join(rows)}
  </table>
</div>
"""


def _build_retrieved_parents(ret: RetrievalSnapshot) -> str:
    """Build the retrieved parent chunks section."""
    cards: List[str] = []
    for rp in ret.retrieved_parents:
        text_display = _esc(rp.text)
        if len(rp.text) > _MAX_INLINE_TEXT:
            text_display = (
                f'<details><summary>Show full text ({len(rp.text):,} chars)</summary>'
                f'<pre>{text_display}</pre></details>'
            )
            preview = _esc(rp.text[:500]) + "…"
        else:
            preview = text_display

        dense_str = f"{rp.dense_score:.4f}" if rp.dense_score is not None else "—"
        sparse_str = f"{rp.sparse_score:.4f}" if rp.sparse_score is not None else "—"

        child_text_display = ""
        if rp.child_text:
            child_escaped = _esc(rp.child_text)
            if len(rp.child_text) > _MAX_INLINE_TEXT:
                child_text_display = (
                    f'<details><summary>Show matching child text ({len(rp.child_text):,} chars)</summary>'
                    f'<pre>{child_escaped}</pre></details>'
                )
            else:
                child_text_display = f'<div class="muted"><strong>Matching child:</strong><pre class="truncate" style="max-height:150px">{child_escaped}</pre></div>'

        path_display = f'<div class="muted mono">📍 Path: {_esc(rp.path)}</div>' if rp.path else ""

        cards.append(f"""
<div class="card">
  <div>
    <strong>Rank #{rp.rank}</strong>
    {_badge(rp.content_type)}
    <span class="muted">| RRF: <strong>{rp.rrf_score:.4f}</strong></span>
    <span class="muted">| Dense: {dense_str}</span>
    <span class="muted">| Sparse: {sparse_str}</span>
    <span class="muted">| Page {rp.page if rp.page is not None else "—"}</span>
  </div>
  <div class="muted">📄 Source: {_esc(rp.source_title)}</div>
  {_breadcrumbs(rp.breadcrumbs)}
  {path_display}
  {child_text_display}
  <pre class="truncate">{preview}</pre>
  {text_display if len(rp.text) > _MAX_INLINE_TEXT else ''}
</div>""")

    return f"""
<div class="card">
  <h3>Retrieved Parent Chunks (6-Step Retrieval Pipeline)</h3>
  <p class="muted">
    {len(ret.retrieved_parents)} parent chunks retrieved via the 6-step pipeline:<br>
    1. Query embedding → 2. Hybrid search ({ret.rerank_input_count} candidates) →
    3. Child text fetch → 4. Cross-encoder rerank ({ret.rerank_output_count} kept) →
    5. Sibling/cross-ref hydration → 6. Merge + dedupe ({ret.final_context_count} final)
  </p>
  {"".join(cards)}
</div>
"""
