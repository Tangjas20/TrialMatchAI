from __future__ import annotations

import json
import os
from typing import Any, Dict


def load_config(config_path: str = "Matcher/config/config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
