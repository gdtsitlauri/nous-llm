"""NOUS — Neural Omnidirectional Understanding System."""
__version__ = "0.1.0"
__author__ = "NOUS Contributors"
__license__ = "MIT"

from .config import NousConfig, DEFAULT_CONFIG
from .model import NousModel, get_model
from .evolve import NousEvolve

__all__ = ["NousConfig", "DEFAULT_CONFIG", "NousModel", "get_model", "NousEvolve"]
