# Makes the repository root importable for pytest (the repo has no packaging yet).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
