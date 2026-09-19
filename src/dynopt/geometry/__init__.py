"""Parametric geometry: design spaces and CAD backends."""

from .base import DesignSpace, DesignVariable, ParametricModel
from .motor_mount import (
    DEFAULT_MOTOR_MASS,
    MOTOR_MOUNT_SPACE,
    MotorMountCAD,
    build_frame,
    motor_node_index,
)

__all__ = [
    "DesignSpace", "DesignVariable", "ParametricModel",
    "MOTOR_MOUNT_SPACE", "build_frame", "motor_node_index",
    "MotorMountCAD", "DEFAULT_MOTOR_MASS",
]
