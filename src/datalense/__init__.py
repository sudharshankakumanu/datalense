"""DataLense - Visualize and browse your large-scale multimodal data."""

__version__ = "0.1.0"

from datalense.db import LenseDB
from datalense.search import nearest, text

__all__ = ["LenseDB", "nearest", "text", "__version__"]
