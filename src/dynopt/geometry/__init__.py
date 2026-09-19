"""Parametric geometry: design spaces and CAD backends."""

from .base import DesignSpace, DesignVariable, ParametricModel
from .motor_mount import (
    DEFAULT_MOTOR_MASS,
    MAX_PLATE_ASPECT,
    MIN_SLENDERNESS,
    MOTOR_MOUNT_SPACE,
    MotorMountCAD,
    build_frame,
    motor_node_index,
    plate_geometry,
    tier0_validity,
)

__all__ = [
    "DesignSpace", "DesignVariable", "ParametricModel",
    "MOTOR_MOUNT_SPACE", "build_frame", "motor_node_index",
    "MotorMountCAD", "DEFAULT_MOTOR_MASS", "tier0_validity", "plate_geometry",
    "MIN_SLENDERNESS", "MAX_PLATE_ASPECT",
]
