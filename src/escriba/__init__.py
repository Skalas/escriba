from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("escriba")
except PackageNotFoundError:  # running from source without an installed dist
    __version__ = "1.0.0"
