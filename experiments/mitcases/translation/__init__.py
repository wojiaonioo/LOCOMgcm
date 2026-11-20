"""The MITgcm-to-LOCOMgcm case translator, split by responsibility; the public entry is
experiments.mitcases.translate."""

# core first: it holds the translator's original import block, so the external modules
# are imported in the order they were before the split
from . import core  # noqa: F401
