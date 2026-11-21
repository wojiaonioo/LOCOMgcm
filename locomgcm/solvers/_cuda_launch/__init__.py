"""Shared CUDA launch-configuration headers.

No Python lives here: ``launch_config.h`` and ``pybind.h`` are ``#include``d by
every generated ``extension.cpp``/``kernel.cu``.  The package marker exists so
setuptools' ``packages.find`` sees the directory and the ``package-data`` rule
in ``pyproject.toml`` actually ships those headers inside an installed wheel --
without it the JIT loaders compile in a source tree and fail in a wheel.
"""
