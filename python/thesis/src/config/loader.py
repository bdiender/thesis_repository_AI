from typing import Any, Dict
import yaml

def load_config(file_path: str, name: str) -> Dict[str, Any]:
    with open(file_path) as f:
        cfg = yaml.safe_load(f)

    return cfg[name]
