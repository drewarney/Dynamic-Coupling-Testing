r"""A cantilever tube carrying a motor at its tip -- the drone-arm archetype.

Second part in the library, and the reason it exists is to keep the pipeline
honest: it shares no code with the motor mount, has different mode families
(``bending`` / ``axial`` / ``torsion``, with no payload *on* supports), and is
excited by propeller orders rather than a motor's 1X. If anything in the core
still assumed the mount, this part would not run.

Physically it is the standard multirotor arm: a tube clamped at the fuselage
with a motor and propeller at the free end. The first bending pair is almost
always the design driver, and the propeller's blade-pass order sweeps right
through it over the throttle range.
"""

from __future__ import annotations

import numpy as np

from ..dynamics.modal import ModeFamilySpec
from ..fea.reference import MATERIALS, FrameModel, Material, Section
from .base import DesignSpace, DesignVariable, PartSpec

#: Design space, SI units.
CANTILEVER_ARM_SPACE = DesignSpace(
    [
        DesignVariable("length", 0.080, 0.350, 0.200, "m", "clamped-to-motor length"),
        DesignVariable("dia", 0.008, 0.030, 0.016, "m", "tube outer diameter"),
        DesignVariable("thk", 0.0006, 0.0040, 0.0015, "m", "wall thickness"),
    ]
)

#: Motor plus propeller at the tip.
DEFAULT_TIP_MASS = 0.085  # kg
DEFAULT_TIP_INERTIA = (2.0e-5, 4.0e-5, 4.0e-5)  # kg.m^2

#: Smallest bore that is manufacturable and meshable; below it the tube is solid.
MIN_BORE = 0.0004

#: Mode families about the tip. The arm runs along +x, so bending couples each
#: transverse translation with the rotation in its own plane -- ``uy`` with
#: ``rz``, ``uz`` with ``ry``. Grouping all four keeps the two bending branches
#: from being split by an argmax over components that are inherently mixed.
CANTILEVER_ARM_FAMILIES = ModeFamilySpec(
    dofs={"bending": (1, 2, 4, 5), "axial": (0,), "torsion": (3,)},
    branches={"bending": 2},
)


def bore_radius(dia: float, thk: float) -> float:
    """Inner radius, or 0.0 when the wall is thick enough to call it solid."""
    bore = 0.5 * float(dia) - float(thk)
    return bore if bore >= MIN_BORE else 0.0


def build_frame(
    params: dict[str, float],
    material: Material | str = "CFRP_QI",
    tip_mass: float = DEFAULT_TIP_MASS,
    tip_inertia: tuple[float, float, float] = DEFAULT_TIP_INERTIA,
    n_elem: int = 12,
) -> FrameModel:
    """Beam idealisation: a clamped tube with a lumped tip mass.

    Carbon tube by default, which is what multirotor arms are actually made of
    and what makes the bending-vs-mass trade interesting.
    """
    mat = MATERIALS[material] if isinstance(material, str) else material
    L = float(params["length"])
    bore = bore_radius(params["dia"], params["thk"])
    dia = float(params["dia"])
    sec = Section.tube(dia, 0.5 * dia - bore) if bore > 0 else Section.circle(dia)

    nodes = np.zeros((n_elem + 1, 3))
    nodes[:, 0] = np.linspace(0.0, L, n_elem + 1)
    return FrameModel(
        nodes=nodes,
        elements=[(i, i + 1) for i in range(n_elem)],
        sections=[sec] * n_elem,
        material=mat,
        fixed_nodes=[0],
        point_masses={n_elem: tip_mass},
        point_inertias={n_elem: tip_inertia},
    )


class CantileverArm(PartSpec):
    """Drone arm: cantilever tube, motor at the tip."""

    name = "cantilever_arm"

    def __init__(
        self,
        material: str = "CFRP_QI",
        tip_mass: float = DEFAULT_TIP_MASS,
        tip_inertia: tuple[float, float, float] = DEFAULT_TIP_INERTIA,
        boss_dia: float = 0.024,
        boss_len: float = 0.012,
    ):
        self.material = material
        self.tip_mass = tip_mass
        self.tip_inertia = tip_inertia
        self.boss_dia = boss_dia
        self.boss_len = boss_len

    @property
    def design_space(self) -> DesignSpace:
        return CANTILEVER_ARM_SPACE

    @property
    def mode_families(self) -> ModeFamilySpec:
        return CANTILEVER_ARM_FAMILIES

    def build_frame(self, params: dict[str, float]) -> FrameModel:
        return build_frame(
            params,
            material=self.material,
            tip_mass=self.tip_mass,
            tip_inertia=self.tip_inertia,
        )

    def build_cad(self, params: dict[str, float]):
        import cadquery as cq

        mm = 1000.0
        L = float(params["length"]) * mm
        dia = float(params["dia"]) * mm
        bore = bore_radius(params["dia"], params["thk"]) * mm
        boss_d = max(self.boss_dia * mm, dia + 4.0)

        tube = cq.Workplane("YZ").circle(dia / 2)
        if bore > 0:
            tube = tube.circle(bore)
        tube = tube.extrude(L)

        # Motor boss at the free end, and a clamped collar at the root: both are
        # where a real arm carries load, and both keep the mesh away from a
        # knife-edge tube end.
        boss = (
            cq.Workplane("YZ")
            .workplane(offset=L - self.boss_len * mm)
            .circle(boss_d / 2)
            .extrude(self.boss_len * mm)
        )
        collar = cq.Workplane("YZ").circle(boss_d / 2).extrude(self.boss_len * mm)
        return tube.union(boss).union(collar)

    def solver_hints(self, params: dict[str, float]) -> dict:
        """Clamp the root collar face at x = 0, motor smeared over the tip boss."""
        return {
            "fix_selector": "x_plane",
            "fix_plane": 0.0,
            "fix_tol": 1e-3,
            "payload_mass": self.tip_mass * 1e-3,   # mm / tonne / s
            "payload_radius": max(self.boss_dia * 1000.0 * 0.6, 6.0),
        }


#: The part, ready to hand to a solver.
CANTILEVER_ARM = CantileverArm()
