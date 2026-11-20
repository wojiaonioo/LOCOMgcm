"""SCS_CLM (South China Sea 1/24 deg climatology) case-private adapters.

This package holds ingest code that is specific to the MITgcm SCS_CLM
reference case (``references/SCS_CLM``) and deliberately lives
outside the ``defgcm`` trunk: the case reads MIT-native big-endian
binaries whose precision / layout differ from the trunk's generic MDS
readers.

Modules:
  - ``mit_native_io``: faithful readers for MIT-native binaries and
    Fortran namelists (no sign / unit conversion whatsoever).
"""
