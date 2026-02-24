"""
Multimodal indexer implementation.
"""

import sys
from pathlib import Path

# Add parent directory for legacy imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Import the legacy multimodal indexing module
import multimodal_indexing as legacy_multimodal_indexing
