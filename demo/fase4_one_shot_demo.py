# Learning principle: one-shot categorization relies on structural memory,
# allowing immediate concept acquisition without iterative training.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lqnn.demo.fase4_one_shot_demo import run


if __name__ == "__main__":
    run()
