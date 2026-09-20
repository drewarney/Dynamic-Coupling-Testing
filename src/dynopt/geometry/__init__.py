"""Parametric geometry: design spaces and CAD backends."""

from .base import DesignSpace, DesignVariable, ParametricModel, PartSpec
from .registry import PART_NAMES, available_parts, get_part
from .cantilever_arm import (
    CANTILEVER_ARM,
    CANTILEVER_ARM_FAMILIES,
    CANTILEVER_ARM_SPACE,
    CantileverArm,
)
from .motor_mount import (
    DEFAULT_MOTOR_MASS,
    MODE_FAMILIES,
    MOTOR_MOUNT,
    MOTOR_MOUNT_FAMILIES,
    MotorMount,
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
    "DesignSpace", "DesignVariable", "ParametricModel", "PartSpec",
    "get_part", "available_parts", "PART_NAMES",
    "MOTOR_MOUNT_SPACE", "build_frame", "motor_node_index",
    "MotorMountCAD", "MotorMount", "MOTOR_MOUNT", "MOTOR_MOUNT_FAMILIES",
    "MODE_FAMILIES", "DEFAULT_MOTOR_MASS",
    "CANTILEVER_ARM", "CantileverArm", "CANTILEVER_ARM_SPACE",
    "CANTILEVER_ARM_FAMILIES", "tier0_validity", "plate_geometry",
    "MIN_SLENDERNESS", "MAX_PLATE_ASPECT",
]
