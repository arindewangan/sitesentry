"""pytest path bootstrap: make ``aws`` and ``agent`` importable from tests."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
