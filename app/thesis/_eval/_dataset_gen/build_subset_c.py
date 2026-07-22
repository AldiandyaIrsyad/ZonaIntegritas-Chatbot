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
from app.thesis._eval._dataset_gen.concordance import BlindInjectionTracker
from app.thesis._eval._dataset_gen.generator import DatasetGenerator
from app.thesis._eval._dataset_gen.panel import EvaluatorPanel

logger = structlog.get_logger(__name__)

SUBTYPES = [
    # (subtype, expected_label, count) — matches skripsi Tabel 3.5
    ("direct_upi", "in_domain", 15),
    ("indirect_upi", "in_domain", 15),
    ("near_miss_government", "out_of_domain", 10),
    ("adjacent_legal", "out_of_domain", 10),
    ("off_topic", "out_of_domain", 10),
]

# In-domain = UPI internal legal/regulatory documents published via JDIH.
DOMAIN_DESC = (
    "the internal legal/regulatory documents of Universitas Pendidikan "
    "Indonesia (UPI), published via its JDIH portal (Peraturan Rektor, SK "
    "Rektor, Statuta UPI, keputusan Senat Akademik / MWA, internal pedoman)"
)

GENERATOR_SYSTEM_PROMPT = f"""\
You are a dataset generator for a boundary relevance benchmark about \
{DOMAIN_DESC}. Generate queries in Indonesian. Output each item as a JSON \
object on its own line (JSONL) with:
- "query": The query text
- "label": Either "in_domain" or "out_of_domain"
- "subtype": The boundary subtype

Do not include markdown code fences. Output one JSON object per line.
"""

VALIDATION_PROMPT = f"""\
You are evaluating a boundary relevance query for a benchmark.

Query: {{query}}
Claimed Label: {{label}}
Subtype: {{subtype}}

Evaluate:
1. Is the label correct? (in_domain = about {DOMAIN_DESC}; out_of_domain = not)
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
        logger.error("datagen.subset_c.missing_api_key")
        sys.exit(1)

    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker()

    accepted_items: List[Dict[str, str]] = []
    total_generated = 0
    total_rejected = 0

    # Safety bound on retry batches per subtype, in case acceptance rate for a
    # hard boundary subtype (e.g. near_miss_government) is persistently low —
    # mirrors build_subset_a's max_cycles retry design.
    max_batches_per_subtype = 6

    try:
        for subtype, expected_label, per_subtype_count in SUBTYPES:
            if len(accepted_items) >= count:
                break

            subtype_target = min(per_subtype_count, count - len(accepted_items))
            accepted_for_subtype = 0
            batch_num = 0

            seed_prompt_base = f"Subtype '{subtype}' means: "
            if subtype == "direct_upi":
                seed_prompt_base += (
                    "questions that explicitly discuss UPI internal regulation topics: "
                    "isi Peraturan Rektor / SK Rektor, Statuta UPI, tugas Senat Akademik / MWA, "
                    "prosedur akademik yang diatur dalam dokumen internal UPI."
                )
            elif subtype == "indirect_upi":
                seed_prompt_base += (
                    "questions about UPI internal regulations with non-standard formulation: "
                    "colloquial language, ambiguous references, no explicit document names."
                )
            elif subtype == "near_miss_government":
                seed_prompt_base += (
                    "government or other-university regulation topics that share vocabulary with "
                    "UPI's documents but are NOT UPI's internal documents (not in the knowledge base)."
                )
            elif subtype == "adjacent_legal":
                seed_prompt_base += (
                    "general Indonesian legal/regulatory questions outside UPI's internal scope "
                    "(e.g. UU ASN, procurement law, national education law not specific to UPI)."
                )
            else:  # off_topic
                seed_prompt_base += "unrelated questions formulated in a formal bureaucratic tone."

            while (
                accepted_for_subtype < subtype_target
                and len(accepted_items) < count
                and batch_num < max_batches_per_subtype
            ):
                n = subtype_target - accepted_for_subtype
                logger.info(
                    "datagen.subset_c.generating",
                    subtype=subtype,
                    batch_size=n,
                    batch_num=batch_num + 1,
                    expected_label=expected_label,
                    have=accepted_for_subtype,
                    target=subtype_target,
                )

                seed_prompt = f"Generate {n} boundary relevance queries of subtype '{subtype}'.\n{seed_prompt_base}"

                try:
                    drafts = await generator.generate(
                        seed_prompt=seed_prompt,
                        count=n,
                        system_prompt=GENERATOR_SYSTEM_PROMPT,
                    )
                except Exception as e:
                    logger.error("datagen.subset_c.generator_error", error=str(e), exc_info=True)
                    batch_num += 1
                    continue

                total_generated += len(drafts)

                for draft in drafts:
                    if accepted_for_subtype >= subtype_target or len(accepted_items) >= count:
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
                        logger.error("datagen.subset_c.panel_error", error=str(e), exc_info=True)
                        continue

                    if verdict.accepted:
                        row = {
                            "query": query,
                            "label": label,
                            "subtype": stype,
                        }
                        accepted_items.append(row)
                        accepted_for_subtype += 1
                        # Track 5/5-unanimous items for blind injection
                        if verdict.yes_count == len(verdict.votes):
                            blind_tracker.add_candidate({**row, "_panel_yes": verdict.yes_count})
                        logger.info("datagen.subset_c.accepted", accepted=len(accepted_items), target=count)
                    else:
                        total_rejected += 1
                        logger.info("datagen.subset_c.rejected", yes=verdict.yes_count, total=verdict.no_count + verdict.yes_count)

                batch_num += 1

            if accepted_for_subtype < subtype_target:
                logger.warning(
                    "datagen.subset_c.subtype_underfilled",
                    subtype=subtype,
                    accepted=accepted_for_subtype,
                    target=subtype_target,
                )

    finally:
        await generator.aclose()
        await panel.aclose()

    # Write blind-injection sidecar
    blind_tracker.write_sidecar(
        output_path.replace(".csv", "_blind_injection.csv"),
        fieldnames=["query", "label", "subtype"],
    )

    # Write CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "label", "subtype"])
        writer.writeheader()
        writer.writerows(accepted_items)

    logger.info(
        "datagen.subset_c.complete",
        generated=total_generated,
        accepted=len(accepted_items),
        rejected=total_rejected,
        output=output_path,
    )


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
