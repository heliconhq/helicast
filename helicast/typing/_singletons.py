import threading

__all__ = ["UnsetType", "UNSET"]


class ThreadSafeSingletonMetaclass(type):
    """Thread safe Singleton metaclass. Usage:

    .. code:: python

    class MySingletonName(metaclass=SingletonMetaclass):
        pass
    """

    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(
                        ThreadSafeSingletonMetaclass, cls
                    ).__call__(*args, **kwargs)
        return cls._instances[cls]


class UnsetType(metaclass=ThreadSafeSingletonMetaclass):
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "<UNSET>"


# Initialize a singleton
UNSET = UnsetType()
