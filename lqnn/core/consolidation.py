# Biological principle: sleep-like consolidation reinforces strong pathways
# and prunes weak or inactive structures.

from __future__ import annotations

from lqnn.core.network import LiquidNetwork


class SleepConsolidator:
    """Encapsulates sleep-cycle consolidation over a living network."""

    def __init__(self, network: LiquidNetwork) -> None:
        self.network = network

    def run(self, sleep_cycles: int = 3) -> None:
        """Apply sleep-like consolidation to reinforce and prune topology."""
        self.network.consolidate(sleep_cycles=sleep_cycles)
