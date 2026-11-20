"""Eager-import of all built-in Layer B sources.

Importing this package triggers `@register_source(name)` decorators on each
module's plugin class, populating the DATA_SOURCES registry.
"""

from . import analytical  # noqa: F401
from . import analytical_h480  # noqa: F401
from . import analytical_iw  # noqa: F401
from . import era5  # noqa: F401
from . import gebco  # noqa: F401
from . import gong_obcs  # noqa: F401
from . import hycom  # noqa: F401
from . import hycom_raw  # noqa: F401
from . import mit_input_binary  # noqa: F401
from . import tpxo9  # noqa: F401
from . import zarr_bundle  # noqa: F401
from . import zero_increment  # noqa: F401

__all__: list[str] = []
