from . import fileio
from . import image_operations
from . import metrics
from . import viewer

# this is needed to get the package version at runtime
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
