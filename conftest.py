# conftest.py — pytest configuration for the project root
# Ensures the project root is on sys.path so imports work correctly
# regardless of where pytest is invoked from.
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
