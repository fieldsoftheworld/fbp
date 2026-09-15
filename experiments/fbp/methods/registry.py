"""The strategy-id keyword -> Method class table, populated by each rule module."""

from collections.abc import Callable

from fbp.methods.base import Method

REGISTRY: dict[str, type[Method]] = {}


def register[M: type[Method]](*keywords: str) -> Callable[[M], M]:
    """Class decorator: registers a ``Method`` subclass under one or more keywords."""

    def deco(cls: M) -> M:
        for kw in keywords:
            REGISTRY[kw] = cls
        return cls

    return deco
