"""Build Subset C — Boundary Relevance.

Generates boundary relevance queries (in-domain and out-of-domain) using
the Generator-Evaluator architecture.

Skripsi §3.2.1, Tabel 3.5.

Pipeline:
    1. Generator (DeepSeek V4) produces draft boundary queries per subtype
    2. Panel (5 models) labels each query as in_domain or out_of_domain
    3. Accept if ≥4/5 panel members agree on the label
    4. Output CSV matching Tabel 3.5 schema

Usage:
    python -m app.thesis._eval._dataset_gen.build_subset_c \\
        --output data/subset_c.csv \\
        --count 60
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from typing import Dict, List

import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from app.thesis._eval._dataset_gen.generator import DatasetGenerator
from app.thesis._eval._dataset_gen.panel import EvaluatorPanel

logger = structlog.get_logger(__name__)

SUBTYPES = [
    ("direct_factual", "in_domain", 10),
    ("procedural", "in_domain", 10),
    ("complex", "in_domain", 10),
    ("off_topic", "out_of_domain", 10),
    ("adjacent", "out_of_domain", 10),
    ("ambiguous", "out_of_domain", 10),
]

GENERATOR_SYSTEM_PROMPT = """\
You are a dataset generator for a boundary relevance benchmark about \
Indonesian bureaucratic reform (Zona Integritas / ZI). Generate queries \
in Indonesian. Output each item as a JSON object on its own line (JSONL) with:
- "query": The query text
- "label": Either "in_domain" or "out_of_domain"
- "subtype": The boundary subtype

Do not include markdown code fences. Output one JSON object per line.
"""

VALIDATION_PROMPT = """\
You are evaluating a boundary relevance query for a benchmark.

Query: {query}
Claimed Label: {label}
Subtype: {subtype}

Evaluate:
1. Is the label correct? (in_domain = about ZI/bureaucratic reform, out_of_domain = not)
2. Is the subtype appropriate?
3. Is the query realistic and well-crafted?

Answer with ONLY 'YES' or 'NO'.
"""


async def build_subset_c(
    settings: DatasetGenSettings,
    output_path: str,
    count: int,
) -> None:
    """Build Subset C (boundary relevance queries) and save to CSV.

    Args:
        settings: Dataset generation settings.
        output_path: Path to output CSV file.
        count: Target number of accepted items.
    """
    if not settings.openrouter_api_key:
        print("ERROR: DATAGEN_OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)

    accepted_items: List[Dict[str, str]] = []
    total_generated = 0
    total_rejected = 0

    try:
        for subtype, expected_label, per_subtype_count in SUBTYPES:
            if len(accepted_items) >= count:
                break

            n = min(per_subtype_count, count - len(accepted_items))
            print(f"\nGenerating {n} {subtype} queries (expected: {expected_label})...")

            seed_prompt = (
                f"Generate {n} boundary relevance queries of subtype '{subtype}'.\n"
                f"Subtype '{subtype}' means: "
            )
            if subtype == "direct_factual":
                seed_prompt += "direct factual questions clearly within the ZI domain."
            elif subtype == "procedural":
                seed_prompt += "procedural questions within the ZI domain."
            elif subtype == "complex":
                seed_prompt += "complex multi-hop questions within the ZI domain."
            elif subtype == "off_topic":
                seed_prompt += "questions completely unrelated to ZI or bureaucratic reform."
            elif subtype == "adjacent":
                seed_prompt += "questions adjacent to ZI but not covered by the KB (e.g., other countries' anti-corruption)."
            else:  # ambiguous
                seed_prompt += "ambiguous questions that could be interpreted as in-domain but are actually out-of-domain."

            try:
                drafts = await generator.generate(
                    seed_prompt=seed_prompt,
                    count=n,
                    system_prompt=GENERATOR_SYSTEM_PROMPT,
                )
            except Exception as e:
                print(f"  Generator error: {e}", file=sys.stderr)
                continue

            total_generated += len(drafts)

            for draft in drafts:
                if len(accepted_items) >= count:
                    break

                if not isinstance(draft.parsed, dict):
                    continue

                item = draft.parsed
                query = item.get("query", "").strip()
                if not query:
                    continue

                label = item.get("label", expected_label).lower().strip()
                stype = item.get("subtype", subtype).lower().strip()

                validation_context = VALIDATION_PROMPT.format(
                    query=query,
                    label=label,
                    subtype=stype,
                )

                try:
                    verdict = await panel.evaluate(
                        prompt="Is this boundary relevance query correctly labeled and well-crafted?",
                        context=validation_context,
                    )
                except Exception as e:
                    print(f"  Panel error: {e}", file=sys.stderr)
                    continue

                if verdict.accepted:
                    accepted_items.append({
                        "query": query,
                        "label": label,
                        "subtype": stype,
                    })
                    print(f"  ✓ Accepted ({len(accepted_items)}/{count})")
                else:
                    total_rejected += 1
                    print(f"  ✗ Rejected ({verdict.yes_count}/{verdict.no_count + verdict.yes_count})")

    finally:
        await generator.aclose()
        await panel.aclose()

    # Write CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "label", "subtype"])
        writer.writeheader()
        writer.writerows(accepted_items)

    print(f"\n{'=' * 60}")
    print(f"Subset C generation complete:")
    print(f"  Generated: {total_generated}")
    print(f"  Accepted:  {len(accepted_items)}")
    print(f"  Rejected:  {total_rejected}")
    print(f"  Output:    {output_path}")
    print(f"{'=' * 60}")


def main() -> None:
    """Entry point for Subset C generation."""
    parser = argparse.ArgumentParser(
        description="Build Subset C (Boundary Relevance) using Generator-Evaluator architecture."
    )
    parser.add_argument(
        "--output",
        default="data/subset_c.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=60,
        help="Target number of accepted items",
    )
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    asyncio.run(build_subset_c(settings, args.output, args.count))


if __name__ == "__main__":
    main()
