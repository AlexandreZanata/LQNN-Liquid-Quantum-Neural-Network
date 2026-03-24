# Scientific principle: automated stimulation/sleep cycles reveal structural
# adaptation rates and long-term network stability.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lqnn.experiments.auto_runner import run


if __name__ == "__main__":
    run()
