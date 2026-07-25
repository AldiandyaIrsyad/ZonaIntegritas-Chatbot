"""Build Subset C — Boundary Relevance.

Generates boundary relevance queries (in-domain and out-of-domain) using
the Generator-Evaluator architecture.

Pipeline:
    1. Generator produces draft boundary queries per subtype
    2. Panel labels each query as in_domain or out_of_domain
    3. Accept if ≥4/5 panel members agree on the label
    4. Output CSV with columns: query, label, subtype, panel_yes, panel_size
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from app.thesis._eval._dataset_gen.checkpoint import IncrementalCSVWriter, resume_rows
from app.thesis._eval._dataset_gen.concordance import BlindInjectionTracker
from app.thesis._eval._dataset_gen.generator import DatasetGenerator
from app.thesis._eval._dataset_gen.panel import EvaluatorPanel, PanelUnavailableError
from app.thesis._eval._dataset_gen.provenance import write_provenance

logger = structlog.get_logger(__name__)

SUBTYPES = [
    # (subtype, expected_label, count) — total 200, balanced 100 in / 100 out.
    #
    # Every subtype clears roughly ±14pp (the in-domain pair clears ±11pp), so
    # per-subtype conclusions are possible. The in/out split is held at 100/100
    # because Exp1b reports Accuracy over the whole set: an uneven split would
    # let the majority class drive the headline number without that being
    # visible in the table.
    ("direct_upi", "in_domain", 50),
    ("indirect_upi", "in_domain", 50),
    ("near_miss_government", "out_of_domain", 34),
    ("adjacent_legal", "out_of_domain", 33),
    ("off_topic", "out_of_domain", 33),
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
    resume: bool = False,
) -> None:
    """Build Subset C (boundary relevance queries) and save to CSV.

    Args:
        resume: Continue an interrupted run, keeping rows already written to
            ``output_path`` and rebuilding the per-subtype counters from them.
    """
    if not settings.openrouter_api_key:
        logger.error("datagen.subset_c.missing_api_key")
        sys.exit(1)

    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker(min_count=20)

    accepted_items: List[Dict[str, str]] = list(
        resume_rows(output_path, ["query", "label", "subtype", "panel_yes", "panel_size"]) if resume else []
    )
    total_generated = 0
    total_rejected = 0

    # Safety bound on retry batches per subtype, in case acceptance rate for a
    # hard boundary subtype (e.g. near_miss_government) is persistently low —
    # mirrors build_subset_a's max_cycles retry design.
    max_batches_per_subtype = 25

    # Two-pass acceptance. Pass 1 uses the strict >= acceptance_threshold rule.
    # Only subtypes still short after exhausting their batch budget get a
    # pass 2, which admits items one vote below the threshold.
    #
    # Why: near_miss_government is *defined* as borderline ("shares vocabulary
    # with UPI's documents but is not one of them"), so a strict >=4/5 rule
    # fights the subtype's own definition — the panel genuinely cannot reach
    # 4/5 on items whose whole purpose is to sit near the boundary, which would
    # silently starve the most informative subtype.
    #
    # Panel agreement is recorded per row as ``panel_yes`` rather than baked
    # into a boolean, so any threshold can be applied after the fact and Exp1b
    # can report the strict and contested slices separately.
    relaxed_threshold = max(1, settings.acceptance_threshold - 1)
    strict_batches = max(1, max_batches_per_subtype // 2)

    # Rows are flushed as they are accepted so an interrupted run leaves a
    # valid partial dataset that --resume can continue.
    writer_ctx = IncrementalCSVWriter(
        output_path, ["query", "label", "subtype", "panel_yes", "panel_size"], resume=resume
    )
    try:
        with writer_ctx as row_writer:
            for subtype, expected_label, per_subtype_count in SUBTYPES:
                if len(accepted_items) >= count:
                    break

                # Seeded from resumed rows so a continued run tops up the shortfall.
                accepted_for_subtype = sum(
                    1 for r in accepted_items if r.get("subtype") == subtype
                )
                subtype_target = min(
                    per_subtype_count, count - len(accepted_items) + accepted_for_subtype
                )
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
                    # Pass 1 (strict) owns the first half of the batch budget. A
                    # subtype still short after that enters pass 2, which admits
                    # items one vote below the threshold. Easy subtypes therefore
                    # fill entirely from strictly-validated items and never reach
                    # pass 2; only a subtype the panel genuinely cannot agree on
                    # relaxes — which is the intended behaviour, since that is a
                    # property of the boundary, not of the candidate.
                    effective_threshold = (
                        settings.acceptance_threshold
                        if batch_num < strict_batches
                        else relaxed_threshold
                    )
                    n = subtype_target - accepted_for_subtype
                    logger.info(
                        "datagen.subset_c.generating",
                        subtype=subtype,
                        batch_size=n,
                        batch_num=batch_num + 1,
                        threshold=effective_threshold,
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
                        except PanelUnavailableError:
                            # The API is down, not this candidate. Propagating stops the
                            # run with its output intact instead of burning the batch
                            # budget marking every item rejected; --resume continues it.
                            raise
                        except Exception as e:
                            logger.error("datagen.subset_c.panel_error", error=str(e), exc_info=True)
                            continue

                        if verdict.yes_count >= effective_threshold:
                            row = {
                                "query": query,
                                "label": label,
                                "subtype": stype,
                                # Recorded per row so any threshold can be applied
                                # after the fact: "contested" stays a derived
                                # predicate rather than a decision baked into the
                                # data at generation time.
                                "panel_yes": verdict.yes_count,
                                "panel_size": len(verdict.votes),
                            }
                            accepted_items.append(row)
                            row_writer.append(row)
                            accepted_for_subtype += 1
                            # Track 5/5-unanimous items for blind injection
                            if verdict.yes_count == len(verdict.votes):
                                blind_tracker.add_candidate({**row, "_panel_yes": verdict.yes_count})
                            logger.info(
                                "datagen.subset_c.accepted",
                                subtype=subtype,
                                accepted=len(accepted_items),
                                panel_yes=verdict.yes_count,
                                threshold=effective_threshold,
                                contested=verdict.yes_count < settings.acceptance_threshold,
                                target=count,
                            )
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
        fieldnames=["query", "label", "subtype", "panel_yes", "panel_size"],
    )

    # Write CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # CSV already complete — rows were flushed as accepted (IncrementalCSVWriter).

    strict_rows = sum(
        1 for r in accepted_items if int(r["panel_yes"]) >= settings.acceptance_threshold
    )
    write_provenance(
        output_path,
        subset="c",
        settings=settings,
        row_count=len(accepted_items),
        extra={
            "generated": total_generated,
            "rejected": total_rejected,
            "targets": {s: c for s, _, c in SUBTYPES},
            "by_subtype": Counter(r["subtype"] for r in accepted_items),
            "by_label": Counter(r["label"] for r in accepted_items),
            # Two-pass acceptance: how much of the set met the strict rule and
            # how much was admitted one vote below it. Exp1b reports these
            # slices separately, so the number belongs with the data.
            "acceptance": {
                "strict_threshold": settings.acceptance_threshold,
                "relaxed_threshold": relaxed_threshold,
                "strict_rows": strict_rows,
                "contested_rows": len(accepted_items) - strict_rows,
            },
            "panel_yes_distribution": Counter(int(r["panel_yes"]) for r in accepted_items),
        },
    )

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
        default=200,
        help="Target number of accepted items",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: keep the rows already in --output and "
        "generate only the shortfall. Rows are written as they are accepted, so a "
        "run stopped by an outage or Ctrl-C can always be continued this way.",
    )
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    asyncio.run(build_subset_c(settings, args.output, args.count, resume=args.resume))


if __name__ == "__main__":
    main()
