"""
Config package entry point.

Re-exports all symbols from config/config.py so existing imports
(`from config import ...`) keep working after moving the module.
"""
from .config import *  # noqa: F401,F403

