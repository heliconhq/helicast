import logging
from contextlib import contextmanager

__all__ = ["configure_logging"]


def configure_logging():
    """Configure the basic logging format. Can be used in the helicast modules.

    .. code:: python

        from helicast.logging import configure_logging

        configure_logging()
        logger = logging.getLogger(__name__)

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    )
