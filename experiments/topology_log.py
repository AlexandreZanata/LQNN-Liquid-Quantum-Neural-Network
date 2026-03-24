# Scientific principle: log topology evolution to quantify structural memory.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lqnn.experiments.topology_log import run


if __name__ == "__main__":
    run()
