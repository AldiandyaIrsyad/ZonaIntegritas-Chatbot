"""Human-Machine Concordance Rate helper.

Implements the concordance measurement defined in skripsi §3.2.1c:
    - Inject 5/5-consensus items blindly into the researcher's verification queue
    - Compute concordance rate (target ≥95%)
    - If concordance < 95%, regenerate the affected subset

Usage:
    concordance = ConcordanceTracker()
    concordance.record(machine_label="supported", human_label="supported")
    rate = concordance.concordance_rate()
    if rate < 0.95:
        print("WARNING: Concordance below 95%, regenerate subset")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ConcordanceTracker:
    """Tracks human-machine agreement for dataset quality control.

    Attributes:
        pairs: List of (machine_label, human_label) tuples.
        target_rate: Minimum acceptable concordance rate (default 0.95).
    """

    pairs: List[Tuple[str, str]] = field(default_factory=list)
    target_rate: float = 0.95

    def record(self, machine_label: str, human_label: str) -> None:
        """Record a single machine-human label pair.

        Args:
            machine_label: Label assigned by the panel.
            human_label: Label assigned by the human researcher.
        """
        self.pairs.append((machine_label.lower().strip(), human_label.lower().strip()))

    def concordance_rate(self) -> float:
        """Compute the concordance rate.

        Returns:
            Proportion of matching labels.
        """
        if not self.pairs:
            return 0.0
        matches = sum(1 for m, h in self.pairs if m == h)
        return matches / len(self.pairs)

    def is_acceptable(self) -> bool:
        """Check if concordance meets the target rate.

        Returns:
            True if concordance_rate >= target_rate.
        """
        rate = self.concordance_rate()
        acceptable = rate >= self.target_rate
        if not acceptable:
            logger.warning(
                "concordance.below_target",
                rate=rate,
                target=self.target_rate,
                total=len(self.pairs),
            )
        return acceptable

    def disagreement_examples(self) -> List[Tuple[str, str, int]]:
        """Get examples where machine and human disagreed.

        Returns:
            List of (machine_label, human_label, count) tuples.
        """
        from collections import Counter
        disagreements = Counter(
            (m, h) for m, h in self.pairs if m != h
        )
        return [(m, h, c) for (m, h), c in disagreements.most_common()]

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Summary string.
        """
        rate = self.concordance_rate()
        total = len(self.pairs)
        matches = sum(1 for m, h in self.pairs if m == h)
        status = "ACCEPTABLE" if self.is_acceptable() else "BELOW TARGET"

        lines = [
            f"Concordance Rate: {rate:.4f} ({matches}/{total}) — {status}",
            f"Target: {self.target_rate:.2%}",
        ]

        if not self.is_acceptable():
            lines.append("Disagreement examples:")
            for m, h, c in self.disagreement_examples()[:5]:
                lines.append(f"  Machine: {m} → Human: {h} (×{c})")

        return "\n".join(lines)
