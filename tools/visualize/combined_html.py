"""HTML report builder for the combined multi-PDF visualization.

Generates a self-contained HTML file (inline CSS, no external dependencies)
from a :class:`CombinedSnapshot`. The report shows:

    - **Section A**: Ingested documents overview (table of all PDFs with counts)
    - **Section B**: Per-query cross-document retrieval (repeated for each query):
        - B1: Per-doc contribution breakdown (CSS bar chart)
        - B2: 3-mode comparison table (dense/sparse/hybrid)
        - B3: Ranked results with doc color badges
    - **Section C**: Score distribution matrix (queries × docs heatmap)
    - **Section D**: Raw data reference

Designed for thesis presentation: "this is how multiple documents interact
during hybrid retrieval".
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List

from .capture import (
    CombinedSnapshot,
    DocIngestionSummary,
    QueryRetrievalSummary,
    RetrievedParentSnapshot,
    RetrievalSnapshot,
    SearchResultSnapshot,
)

# ── Color palette for document badges (11 distinct colors) ──────────
_DOC_COLORS: List[str] = [
    "#3b82f6",  # blue
    "#22c55e",  # green
    "#f97316",  # orange
    "#a855f7",  # purple
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#eab308",  # yellow
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#84cc16",  # lime
]

# Max text length to display inline before truncating
_MAX_INLINE_TEXT = 2000


def build_combined_html(snapshot: CombinedSnapshot) -> str:
    """Build a complete self-contained HTML document from a combined snapshot.

    Args:
        snapshot: The combined multi-PDF snapshot.

    Returns:
        A full HTML string (with inline CSS, no external dependencies).
    """
    doc_color_map = _build_doc_color_map(snapshot.doc_summaries)

    parts: List[str] = []
    parts.append(_HTML_HEAD)
    parts.append(_build_header(snapshot))
    parts.append(_build_docs_overview(snapshot))
    parts.append(_build_per_query_sections(snapshot, doc_color_map))
    parts.append(_build_score_matrix(snapshot, doc_color_map))
    parts.append(_build_raw_data_ref(snapshot))
    parts.append(_HTML_FOOTER)
    return "\n".join(parts)


def write_combined_html(snapshot: CombinedSnapshot, output_path: str) -> str:
    """Build the combined HTML report and write it to a file.

    Args:
        snapshot: The combined multi-PDF snapshot.
        output_path: Absolute path to write the HTML file.

    Returns:
        The path that was written to.
    """
    html_content = build_combined_html(snapshot)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_content, encoding="utf-8")
    return output_path


# ── Helpers ─────────────────────────────────────────────────────────


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(text)


def _build_doc_color_map(
    doc_summaries: List[DocIngestionSummary],
) -> Dict[str, str]:
    """Assign a unique color to each document title.

    Args:
        doc_summaries: List of per-doc summaries.

    Returns:
        Mapping of doc_title → hex color string.
    """
    color_map: Dict[str, str] = {}
    for i, ds in enumerate(doc_summaries):
        color_map[ds.doc_title] = _DOC_COLORS[i % len(_DOC_COLORS)]
    return color_map


def _doc_badge(doc_title: str, color_map: Dict[str, str]) -> str:
    """Render a document badge with its assigned color.

    Args:
        doc_title: The document title.
        color_map: Mapping of doc_title → hex color.

    Returns:
        HTML span element with the doc badge.
    """
    color = color_map.get(doc_title, "#64748b")
    # Truncate long titles for display
    display = doc_title if len(doc_title) <= 50 else doc_title[:47] + "…"
    return f'<span class="badge" style="background:{color}">{_esc(display)}</span>'


def _breadcrumbs(crumbs: List[str]) -> str:
    """Render breadcrumbs as a muted path string."""
    if not crumbs:
        return ""
    return f'<div class="breadcrumbs">📍 {" &gt; ".join(_esc(c) for c in crumbs)}</div>'


# ── HTML fragments ──────────────────────────────────────────────────

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Combined Multi-PDF Retrieval Visualization</title>
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
  h4 { color: #cbd5e1; margin-top: 16px; margin-bottom: 8px; }
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
    white-space: nowrap;
  }
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
    height: 12px;
    border-radius: 4px;
    vertical-align: middle;
    margin-right: 6px;
  }
  .mono { font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 0.8125em; }
  .muted { color: var(--text-muted); }
  .truncate { max-height: 200px; overflow-y: auto; }
  .bar-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    gap: 8px;
  }
  .bar-label {
    display: inline-block;
    width: 200px;
    font-size: 0.8125em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar-container {
    flex: 1;
    background: #334155;
    border-radius: 4px;
    height: 20px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 4px;
    display: flex;
    align-items: center;
    padding-left: 6px;
    font-size: 0.75em;
    font-weight: 600;
    color: #fff;
  }
  .matrix-cell {
    text-align: center;
    padding: 8px 4px;
    font-size: 0.8125em;
  }
  .matrix-score {
    font-weight: 700;
    font-size: 0.9em;
  }
  .matrix-count {
    font-size: 0.7em;
    color: var(--text-muted);
  }
  .query-block {
    border-left: 3px solid var(--accent);
    padding-left: 16px;
    margin: 24px 0;
  }
</style>
</head>
<body>
"""

_HTML_FOOTER = """
</body>
</html>
"""


def _build_header(snapshot: CombinedSnapshot) -> str:
    """Build the report header with summary stats."""
    total_elements = sum(ds.element_count for ds in snapshot.doc_summaries)
    total_parents = sum(ds.parent_count for ds in snapshot.doc_summaries)
    total_children = sum(ds.child_count for ds in snapshot.doc_summaries)
    total_points = sum(ds.qdrant_point_count for ds in snapshot.doc_summaries)
    total_results = sum(qs.total_results for qs in snapshot.query_summaries)

    return f"""
<h1>🔍 Combined Multi-PDF Retrieval Visualization</h1>
<p class="muted">Generated: {_esc(snapshot.timestamp)}</p>

<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-value">{len(snapshot.doc_summaries)}</div>
    <div class="stat-label">Documents Ingested</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{len(snapshot.queries)}</div>
    <div class="stat-label">Queries Run</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{total_elements}</div>
    <div class="stat-label">Total Parsed Elements</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{total_parents}</div>
    <div class="stat-label">Total Parent Chunks</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{total_children}</div>
    <div class="stat-label">Total Child Chunks</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{total_points}</div>
    <div class="stat-label">Total Qdrant Points</div>
  </div>
  <div class="stat-box">
    <div class="stat-value">{total_results}</div>
    <div class="stat-label">Total Retrieved Results</div>
  </div>
</div>

<div class="card">
  <h3>Shared Resources</h3>
  <table>
    <tr><th>Resource</th><th>Value</th></tr>
    <tr><td>Qdrant Collection</td><td class="mono">{_esc(snapshot.qdrant_collection)}</td></tr>
    <tr><td>SQLite Database</td><td class="mono">{_esc(snapshot.sqlite_path)}</td></tr>
  </table>
</div>
"""


def _build_docs_overview(snapshot: CombinedSnapshot) -> str:
    """Build Section A — ingested documents overview table."""
    rows: List[str] = []
    for i, ds in enumerate(snapshot.doc_summaries, start=1):
        ct_str = ", ".join(f"{k}: {v}" for k, v in sorted(ds.content_type_counts.items()))
        rows.append(f"""<tr>
  <td>{i}</td>
  <td>{_esc(ds.doc_title)}</td>
  <td>{ds.element_count}</td>
  <td>{ds.parent_count}</td>
  <td>{ds.child_count}</td>
  <td>{ds.qdrant_point_count}</td>
  <td>{ds.total_chars:,}</td>
  <td class="muted">{_esc(ct_str)}</td>
</tr>""")

    return f"""
<h2>📥 Section A — Ingested Documents Overview</h2>
<div class="card">
  <p class="muted">
    All {len(snapshot.doc_summaries)} PDFs were ingested into the shared Qdrant collection
    <code>{_esc(snapshot.qdrant_collection)}</code>. Each document's chunks carry a
    <code>doc_id</code> payload, enabling cross-document retrieval.
  </p>
  <table>
    <tr>
      <th>#</th>
      <th>Document Title</th>
      <th>Elements</th>
      <th>Parents</th>
      <th>Children</th>
      <th>Points</th>
      <th>Total Chars</th>
      <th>Content Types</th>
    </tr>
    {"".join(rows)}
  </table>
</div>
"""


def _build_per_query_sections(
    snapshot: CombinedSnapshot,
    doc_color_map: Dict[str, str],
) -> str:
    """Build Section B — per-query cross-document retrieval (repeated per query)."""
    parts: List[str] = []
    parts.append("<h2>🔎 Section B — Per-Query Cross-Document Retrieval</h2>")
    parts.append(
        '<p class="muted">Each query is searched in 3 modes (dense, sparse, hybrid) '
        "across ALL documents. Results show how different documents contribute "
        "to the retrieval.</p>"
    )

    for i, (ret, qs) in enumerate(
        zip(snapshot.retrievals, snapshot.query_summaries), start=1
    ):
        parts.append(f'<div class="query-block">')
        parts.append(f"<h3>Query {i}: \"{_esc(ret.query)}\"</h3>")
        parts.append(_build_doc_contribution(qs, doc_color_map))
        parts.append(_build_mode_comparison(ret))
        parts.append(_build_ranked_results(ret, doc_color_map))
        parts.append("</div>")

    return "\n".join(parts)


def _build_doc_contribution(
    qs: QueryRetrievalSummary,
    doc_color_map: Dict[str, str],
) -> str:
    """Build B1 — per-doc contribution breakdown (CSS bar chart)."""
    if not qs.per_doc_counts:
        return """
<div class="card">
  <h4>B1 — Per-Document Contribution</h4>
  <p class="muted">No results retrieved.</p>
</div>
"""

    max_count = max(qs.per_doc_counts.values())
    bars: List[str] = []
    # Sort by count descending
    sorted_docs = sorted(
        qs.per_doc_counts.items(), key=lambda x: x[1], reverse=True
    )
    for doc_title, count in sorted_docs:
        color = doc_color_map.get(doc_title, "#64748b")
        bar_width = int((count / max_count) * 100) if max_count > 0 else 0
        best_score = qs.per_doc_best_score.get(doc_title, 0.0)
        bars.append(f"""<div class="bar-row">
  <span class="bar-label" title="{_esc(doc_title)}">{_esc(doc_title)}</span>
  <div class="bar-container">
    <div class="bar-fill" style="width:{bar_width}%;background:{color}">
      {count} result(s)
    </div>
  </div>
  <span class="muted mono">best: {best_score:.4f}</span>
</div>""")

    return f"""
<div class="card">
  <h4>B1 — Per-Document Contribution Breakdown</h4>
  <p class="muted">
    Total: {qs.total_results} results from {len(qs.per_doc_counts)} document(s).
    Top doc: <strong>{_esc(qs.top_doc_title)}</strong> (score: {qs.top_doc_score:.4f})
  </p>
  {"".join(bars)}
</div>
"""


def _build_mode_comparison(ret: RetrievalSnapshot) -> str:
    """Build B2 — 3-mode search comparison table."""
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
  <h4>B2 — 3-Mode Comparison (Dense vs Sparse vs Hybrid RRF)</h4>
  <details>
    <summary>📋 Full comparison table ({len(all_chunk_ids)} unique chunks)</summary>
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
  </details>
</div>
"""


def _build_ranked_results(
    ret: RetrievalSnapshot,
    doc_color_map: Dict[str, str],
) -> str:
    """Build B3 — ranked results with doc color badges."""
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
    {_doc_badge(rp.source_title, doc_color_map)}
    <span class="muted">| RRF: <strong>{rp.rrf_score:.4f}</strong></span>
    <span class="muted">| Dense: {dense_str}</span>
    <span class="muted">| Sparse: {sparse_str}</span>
    <span class="muted">| Page {rp.page if rp.page is not None else "—"}</span>
  </div>
  {_breadcrumbs(rp.breadcrumbs)}
  {path_display}
  {child_text_display}
  <pre class="truncate">{preview}</pre>
  {text_display if len(rp.text) > _MAX_INLINE_TEXT else ''}
</div>""")

    if not cards:
        cards.append('<p class="muted">No results retrieved.</p>')

    return f"""
<div class="card">
  <h4>B3 — Ranked Results (Hybrid RRF) with Document Badges</h4>
  <p class="muted">
    {len(ret.retrieved_parents)} parent chunks retrieved. Each result is color-coded
    by its source document, showing how documents interleave in the ranking.
  </p>
  {"".join(cards)}
</div>
"""


def _build_score_matrix(
    snapshot: CombinedSnapshot,
    doc_color_map: Dict[str, str],
) -> str:
    """Build Section C — score distribution matrix (queries × docs heatmap)."""
    doc_titles = [ds.doc_title for ds in snapshot.doc_summaries]

    # Build matrix: query_index × doc_title → (best_score, count)
    matrix: Dict[str, Dict[str, tuple[float, int]]] = {}
    for qs in snapshot.query_summaries:
        row: Dict[str, tuple[float, int]] = {}
        for dt in doc_titles:
            score = qs.per_doc_best_score.get(dt, 0.0)
            count = qs.per_doc_counts.get(dt, 0)
            row[dt] = (score, count)
        matrix[qs.query] = row

    # Find global max score for color intensity
    all_scores = [
        qs.per_doc_best_score.get(dt, 0.0)
        for qs in snapshot.query_summaries
        for dt in doc_titles
    ]
    max_score = max(all_scores) if all_scores else 1.0

    # Build table
    header_cells = "".join(
        f'<th style="writing-mode:vertical-rl;text-orientation:mixed;max-width:120px;overflow:hidden">'
        f'<span title="{_esc(dt)}">{_esc(dt[:40])}{"…" if len(dt) > 40 else ""}</span></th>'
        for dt in doc_titles
    )

    rows: List[str] = []
    for qs in snapshot.query_summaries:
        cells: List[str] = []
        for dt in doc_titles:
            score, count = matrix[qs.query][dt]
            if count > 0:
                # Color intensity based on score
                intensity = score / max_score if max_score > 0 else 0
                # Interpolate from dark (low) to bright (high)
                bg_color = _intensity_color(intensity)
                cells.append(
                    f'<td class="matrix-cell" style="background:{bg_color}">'
                    f'<div class="matrix-score">{score:.4f}</div>'
                    f'<div class="matrix-count">{count} result(s)</div>'
                    f'</td>'
                )
            else:
                cells.append(
                    '<td class="matrix-cell" style="background:#1e293b;color:#475569">'
                    '<div class="matrix-score">—</div>'
                    '<div class="matrix-count">0</div>'
                    '</td>'
                )
        query_short = qs.query if len(qs.query) <= 40 else qs.query[:37] + "…"
        rows.append(
            f'<tr><td style="font-weight:600;max-width:200px">{_esc(query_short)}</td>'
            f'{"".join(cells)}</tr>'
        )

    return f"""
<h2>📊 Section C — Score Distribution Matrix</h2>
<div class="card">
  <p class="muted">
    Heatmap showing the best RRF score (and result count) for each query × document pair.
    Brighter cells indicate higher relevance. This reveals which documents are
    relevant to which queries — and which queries retrieve from multiple documents.
  </p>
  <table>
    <tr>
      <th>Query \\ Document</th>
      {header_cells}
    </tr>
    {"".join(rows)}
  </table>
</div>
"""


def _intensity_color(intensity: float) -> str:
    """Interpolate from dark slate to bright cyan based on intensity (0-1).

    Args:
        intensity: Normalized score (0.0 = lowest, 1.0 = highest).

    Returns:
        CSS rgb() color string.
    """
    # Dark: rgb(30, 41, 59) = #1e293b
    # Bright: rgb(56, 189, 248) = #38bdf8
    r = int(30 + (56 - 30) * intensity)
    g = int(41 + (189 - 41) * intensity)
    b = int(59 + (248 - 59) * intensity)
    return f"rgb({r},{g},{b})"


def _build_raw_data_ref(snapshot: CombinedSnapshot) -> str:
    """Build Section D — raw data reference."""
    return f"""
<h2>📁 Section D — Raw Data Reference</h2>
<div class="card">
  <p class="muted">
    The following raw data files were generated alongside this HTML report:
  </p>
  <table>
    <tr><th>File</th><th>Contents</th></tr>
    <tr><td class="mono">combined_01_documents.json</td><td>Per-document ingestion summaries</td></tr>
    <tr><td class="mono">combined_02_queries.json</td><td>Per-query retrieval summaries</td></tr>
    <tr><td class="mono">combined_03_all_retrieved.json</td><td>Full retrieved parents for all queries</td></tr>
    <tr><td class="mono">combined_04_score_matrix.json</td><td>Query × document score matrix</td></tr>
    <tr><td class="mono">combined_summary.json</td><td>Top-level metadata</td></tr>
    <tr><td class="mono">combined_documents.csv</td><td>Tabular doc summaries</td></tr>
    <tr><td class="mono">combined_queries.csv</td><td>Tabular query × doc pairs</td></tr>
    <tr><td class="mono">combined_retrieved.csv</td><td>All retrieved parents across queries</td></tr>
  </table>
  <p class="muted">
    SQLite DB: <code>{_esc(snapshot.sqlite_path)}</code><br>
    Qdrant collection: <code>{_esc(snapshot.qdrant_collection)}</code>
  </p>
</div>
"""
