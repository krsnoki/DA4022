"""Put the repository root on sys.path so ``Dataloaders`` imports as a namespace package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
