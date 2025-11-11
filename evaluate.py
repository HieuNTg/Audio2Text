"""Root wrapper kept for backwards compatibility.

This file imports the package implementation `src/audiototext.evaluate` and calls its
`main()` function. Prefer to use `python run_evaluate.py` or import from
`src/audiototext` directly.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audiototext import evaluate


if __name__ == "__main__":
    evaluate.main()
