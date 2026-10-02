"""Compatibility import with the existing Habitat logger."""

from habitat import logger
from lavira.clients import LaViRA_API as _SharedAPI


class LaViRA_API(_SharedAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, log=logger, **kwargs)
