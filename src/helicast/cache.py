from functools import partial, wraps
from pathlib import Path

from joblib import Memory, expires_after

__all__ = [
    "CACHE_DIR",
    "MEMORY",
    "clear_cache",
    "cache",
]

CACHE_DIR = Path(Path.home(), ".cache", "helicast")

MEMORY = Memory(location=CACHE_DIR, verbose=0)


def cache(
    func=None,
    *,
    ignore=None,
    verbose=None,
    mmap_mode=False,
    cache_validation_callback=None,
):
    """Decorator to cache function using ``joblib.Memory`` class. The default cache
    location is ``~/.cache/helicast``. The parameters of the decorator are
    passed to the `joblib.Memory` class."""

    if func is None:
        return partial(
            cache,
            ignore=ignore,
            verbose=verbose,
            mmap_mode=mmap_mode,
            cache_validation_callback=cache_validation_callback,
        )

    @wraps(func)
    def wrapped(*args, **kwargs):
        return MEMORY.cache(
            func=func,
            ignore=ignore,
            verbose=verbose,
            mmap_mode=mmap_mode,
            cache_validation_callback=cache_validation_callback,
        )(*args, **kwargs)

    return wrapped


def cache_expires_after(func=None, *, seconds: int):
    """Decorator to cache function using ``joblib.Memory`` class with expiration time.
    This decorator is a wrapper around the `cache` decorator with the `expires_after`
    cache validation callback.

    Args:
        func: Function to decorate. Defaults to None.
        seconds: Number of seconds before the cache expires.

    Returns:
        Wrapped function.
    """
    return cache(func, cache_validation_callback=expires_after(seconds=seconds))


def clear_cache():
    """Clear the cache located at ``~/.cache/helicast``."""
    MEMORY.clear(warn=False)
