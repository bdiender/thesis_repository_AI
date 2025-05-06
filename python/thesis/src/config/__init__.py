import os
import yaml

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
        val = self._data[key]

        if isinstance(val, dict):
            return Config(_data=val)
        return val

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

    def to_dict(self):
        return self._data

GLOBAL_CONFIG = Config()
