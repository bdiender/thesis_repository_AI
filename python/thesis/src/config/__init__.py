import os
import yaml
from pprint import pformat

class Config:
    def __init__(self, path=None, _data=None):
        if _data is not None:
            self._data = _data
            return

        if path is None:
            here = os.path.dirname(__file__)
            path = os.path.abspath(os.path.join(here, 'config.yaml'))
        else:
            path = os.path.abspath(path)

        with open(path) as f:
            self._data = yaml.safe_load(f)

    def __getattr__(self, key):
        try:
            val = self._data[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

        if isinstance(val, dict):
            return Config(_data=val)
        return val
    
    def __repr__(self):
        return f'{self.__class__.__name__}({pformat(self._data)})'

    def get(self, *keys, default=None):
        if len(keys) >= 2 and not isinstance(keys[-1], str):
            default = keys[-1]
            keys = keys[:-1]

        node = self
        for k in keys:
            try:
                node = getattr(node, k)
            except AttributeError:
                return default

        return node
    
    def set(self, key_path: str, value: str):
        parent, key = self._search_leaf(key_path)
        if parent is None:
            return
        
        orig = parent[key]
        try:
            if isinstance(orig, bool):
                parent[key] = value.lower() in ('true', '1')
            else:
                parent[key] = type(orig)(value)
        except Exception:
            parent[key] = value

    def to_dict(self):
        return self._data

    def _search_leaf(self, parameter: str):
        stack = [self._data]

        while stack:
            node = stack.pop()

            for k, v in node.items():
                if k == parameter and not isinstance(v, dict):
                    return node, k
                if isinstance(v, dict):
                    stack.append(v)
        
        return None, None

GLOBAL_CONFIG = Config()
