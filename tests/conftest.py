"""
Shared test configuration.

Ensures the repo root is on sys.path so `finance_tools` is importable
even without pip install.
"""

import os
import sys

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
