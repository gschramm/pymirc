from . import fileio
from . import image_operations
from . import metrics
from . import viewer

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pymirc")
except PackageNotFoundError:
    __version__ = "unknown"
