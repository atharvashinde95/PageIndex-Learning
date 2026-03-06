"""
core/index_store.py
-------------------
Save and load the PageIndex tree JSON to/from the local results/ directory.
Keeps the rest of the code clean by centralising all I/O here.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def _index_path(pdf_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_name).stem
    return RESULTS_DIR / f"{stem}_index.json"


def save_index(tree: Dict[str, Any], pdf_name: str) -> Path:
    """Persist the tree dict to disk as pretty-printed JSON."""
    path = _index_path(pdf_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(tree, fh, indent=2, ensure_ascii=False)
    logger.info("Index saved to %s", path)
    return path


def load_index(pdf_name: str) -> Optional[Dict[str, Any]]:
    """Load a previously saved index. Returns None if not found."""
    path = _index_path(pdf_name)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Index loaded from %s", path)
    return data


def index_exists(pdf_name: str) -> bool:
    return _index_path(pdf_name).exists()
