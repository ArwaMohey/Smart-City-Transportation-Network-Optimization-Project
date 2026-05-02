"""Pytest configuration for the Smart City Transportation Network test suite.

Adds the project root to ``sys.path`` so that absolute imports (e.g.
``from models.node import Node``) work from any working directory.
"""

import sys
from pathlib import Path

# Insert the project root (parent of this conftest.py) at the front of
# sys.path so that absolute package imports resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
