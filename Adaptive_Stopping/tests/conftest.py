import os
import sys

# Make `Adaptive_Stopping/methods` importable so tests can `import tempered_remasking`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "methods"))
