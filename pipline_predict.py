"""Compatibility wrapper for legacy `python pipline_predict.py` runs."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audiototext import pipline_predict


def main() -> None:
    """Delegate to the package implementation."""
    pipline_predict.main()


if __name__ == "__main__":
    main()
