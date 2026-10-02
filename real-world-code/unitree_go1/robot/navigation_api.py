"""Platform configuration for the shared LaViRA reasoning profile."""

from config import Config
from lavira.reasoning.go1 import LaViRANavigationAPI as _SharedAPI


class LaViRANavigationAPI(_SharedAPI):
    def __init__(self, client):
        super().__init__(client, config=Config)
