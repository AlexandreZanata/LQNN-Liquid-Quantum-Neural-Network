# Quantum-inspired principle: evaluate branch efficiency with free-energy-like
# metrics.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lqnn.experiments.energy_profile import run


if __name__ == "__main__":
    run()
