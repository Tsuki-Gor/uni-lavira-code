"""Compatibility import for the shared LaViRA model client."""

from lavira.clients import LaViRAVisionClient as _SharedClient


class LaViRAVisionClient(_SharedClient):
    def __init__(self, config=None):
        if config is None:
            from config import Config
            config = Config
        super().__init__(config)
