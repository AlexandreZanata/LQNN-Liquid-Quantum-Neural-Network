# Biological principle: recurring patterns shape stable activation pathways.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lqnn.demo.pattern_demo import run


if __name__ == "__main__":
    run()
