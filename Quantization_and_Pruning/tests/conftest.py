"""Make the repository root importable so tests can `import Quantization_and_Pruning...`."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "Quantization_and_Pruning" / "configs" / "cafeq"
