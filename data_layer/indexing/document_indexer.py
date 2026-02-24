"""
Document indexer implementation.
"""

import sys
from pathlib import Path

# Add parent directory for legacy imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Import the legacy indexing module
import indexing as legacy_indexing
