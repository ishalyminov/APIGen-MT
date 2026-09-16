"""Make the standalone pass@k scripts importable in their original layout."""

from __future__ import annotations

import sys
from pathlib import Path


PASSK_ROOT = Path(__file__).resolve().parents[1]
if str(PASSK_ROOT) not in sys.path:
    sys.path.insert(0, str(PASSK_ROOT))
