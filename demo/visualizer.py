# Biological principle: observable dynamics are required to validate adaptive
# topology in real time.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lqnn.demo.visualizer import run_live_visualization


if __name__ == "__main__":
    run_live_visualization()
