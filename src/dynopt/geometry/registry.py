"""Registry of available parts.

Lives here, next to the parts, rather than in the CLI: adding a part should
touch exactly one package. The core pipeline never imports this -- it works
against whatever :class:`PartSpec` it is handed.

Imports are lazy so listing parts costs nothing, and a part whose CAD backend is
missing stays listable.
"""

from __future__ import annotations

from typing import Callable

from .base import PartSpec


def _motor_mount() -> PartSpec:
    from .motor_mount import MOTOR_MOUNT

    return MOTOR_MOUNT


def _cantilever_arm() -> PartSpec:
    from .cantilever_arm import CANTILEVER_ARM

    return CANTILEVER_ARM


#: name -> zero-argument factory. Aliases are welcome; keep the canonical name
#: first in :data:`PART_NAMES`.
PART_FACTORIES: dict[str, Callable[[], PartSpec]] = {
    "motor_mount": _motor_mount,
    "mount": _motor_mount,
    "cantilever_arm": _cantilever_arm,
    "arm": _cantilever_arm,
    "drone_arm": _cantilever_arm,
}

#: Canonical part names, for help text and error messages.
PART_NAMES: tuple[str, ...] = ("motor_mount", "cantilever_arm")


def get_part(name: str) -> PartSpec:
    """Resolve a part by name, or raise with the list of known parts."""
    try:
        return PART_FACTORIES[name]()
    except KeyError:
        raise KeyError(
            f"unknown part {name!r}; known parts: {', '.join(PART_NAMES)}"
        ) from None


def available_parts() -> tuple[str, ...]:
    return PART_NAMES
