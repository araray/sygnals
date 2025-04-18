# sygnals/config/__init__.py

"""
Configuration management for the Sygnals application.

This package handles loading configuration from files (TOML),
environment variables, and internal defaults, providing a unified
configuration object.
"""

from .models import SygnalsConfig
from .loaders import load_configuration

__all__ = [
    "SygnalsConfig",
    "load_configuration",
]
