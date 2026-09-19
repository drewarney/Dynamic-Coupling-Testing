r"""The benchmark part: a parametric 4-leg motor mount / bracket.

Physical description
--------------------
A motor of mass :math:`m_{motor}` sits on a square platform carried by four
inclined legs bolted to a base plate.  This is the canonical resonance problem:
the motor's 1X and blade-pass excitation orders sweep through the operating RPM
range and must not intersect any structural mode.

Five design variables control it::

    leg_thk    wall thickness of the tubular legs   [m]
    leg_dia    outer diameter of the legs           [m]
    span       base footprint half-span             [m]
    height     platform height above the base       [m]
    plate_thk  platform plate thickness             [m]

Two representations of the *same* parameters are provided so you can move up
the fidelity ladder without changing the optimiser:

* :func:`build_frame` -- a 1D space-frame idealisation for the reference solver
  (milliseconds per solve, good for DOE and surrogate training);
* :class:`MotorMountCAD` -- a CadQuery solid exported to STEP, meshed and
  solved with CalculiX or MAPDL (seconds to minutes per solve, used to
  validate and re-anchor the surrogate).
"""

from __future__ import annotations

import numpy as np

from ..fea.reference import MATERIALS, FrameModel, Material, Section
from .base import DesignSpace, DesignVariable, ParametricModel

#: Design space for the benchmark motor mount, all dimensions in metres.
MOTOR_MOUNT_SPACE = DesignSpace(
    [
        DesignVariable("leg_thk", 0.0010, 0.0060, 0.0020, "m", "leg wall thickness"),
        DesignVariable("leg_dia", 0.0080, 0.0250, 0.0140, "m", "leg outer diameter"),
        DesignVariable("span", 0.0400, 0.1200, 0.0700, "m", "base half-span"),
        DesignVariable("height", 0.0300, 0.1000, 0.0600, "m", "platform height"),
        DesignVariable("plate_thk", 0.0015, 0.0080, 0.0030, "m", "plate thickness"),
    ]
)

#: Non-structural payload carried at the platform centre.
DEFAULT_MOTOR_MASS = 0.35  # kg
DEFAULT_MOTOR_INERTIA = (2.0e-4, 2.0e-4, 3.5e-4)  # kg.m^2 about x, y, z


def build_frame(
    params: dict[str, float],
    material: Material | str = "AL6061",
    motor_mass: float = DEFAULT_MOTOR_MASS,
    motor_inertia: tuple[float, float, float] = DEFAULT_MOTOR_INERTIA,
    n_seg: int = 4,
) -> FrameModel:
    """Space-frame idealisation of the motor mount.

    Node layout: 4 base corners (fully fixed), 4 platform corners, 1 platform
    centre carrying the motor.  Legs are subdivided into ``n_seg`` elements each
    so bending mode shapes resolve properly -- a single element per leg
    over-stiffens the structure by several percent.
    """
    mat = MATERIALS[material] if isinstance(material, str) else material

    thk = float(params["leg_thk"])
    dia = float(params["leg_dia"])
    span = float(params["span"])
    h = float(params["height"])
    pthk = float(params["plate_thk"])

    # Platform is a fixed fraction of the base footprint (tapered legs).
    top_span = 0.55 * span

    leg_sec = Section.tube(dia, min(thk, dia / 2.0 - 1e-5))
    # The plate is modelled as four edge beams of equivalent bending stiffness:
    # width = the tributary half-span, depth = the plate thickness.
    plate_sec = Section.rectangle(max(top_span, 1e-3), pthk)
    # Spokes from the platform corners to the motor node, same section.
    spoke_sec = Section.rectangle(max(0.5 * top_span, 1e-3), pthk)

    corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    nodes, elements, sections = [], [], []

    # --- legs: base corner -> platform corner, n_seg elements each ----------
    top_idx = []
    for cx, cy in corners:
        p0 = np.array([cx * span, cy * span, 0.0])
        p1 = np.array([cx * top_span, cy * top_span, h])
        start = len(nodes)
        for s in range(n_seg + 1):
            nodes.append(p0 + (p1 - p0) * (s / n_seg))
        for s in range(n_seg):
            elements.append((start + s, start + s + 1))
            sections.append(leg_sec)
        top_idx.append(start + n_seg)

    # --- platform perimeter --------------------------------------------------
    for a in range(4):
        elements.append((top_idx[a], top_idx[(a + 1) % 4]))
        sections.append(plate_sec)

    # --- motor node + spokes -------------------------------------------------
    motor_node = len(nodes)
    nodes.append(np.array([0.0, 0.0, h]))
    for t in top_idx:
        elements.append((t, motor_node))
        sections.append(spoke_sec)

    base_nodes = [i * (n_seg + 1) for i in range(4)]

    return FrameModel(
        nodes=np.array(nodes),
        elements=elements,
        sections=sections,
        material=mat,
        fixed_nodes=base_nodes,
        point_masses={motor_node: motor_mass},
        point_inertias={motor_node: motor_inertia},
    )


def motor_node_index(n_seg: int = 4) -> int:
    """Index of the motor node in :func:`build_frame` output."""
    return 4 * (n_seg + 1)


class MotorMountCAD(ParametricModel):
    """CadQuery solid model of the same part, exported to STEP.

    Imported lazily so the package stays usable without the (large) CadQuery /
    OCP dependency -- the reference tier needs only numpy and scipy.
    """

    def __init__(self, motor_dia: float = 0.028, bolt_dia: float = 0.004):
        self.motor_dia = motor_dia
        self.bolt_dia = bolt_dia

    @property
    def design_space(self) -> DesignSpace:
        return MOTOR_MOUNT_SPACE

    def build(self, params: dict[str, float]):
        try:
            import cadquery as cq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "MotorMountCAD needs CadQuery: pip install 'dynopt[oss]'"
            ) from exc

        span = float(params["span"])
        h = float(params["height"])
        pthk = float(params["plate_thk"])
        dia = float(params["leg_dia"])
        thk = float(params["leg_thk"])
        top_span = 0.55 * span

        # CadQuery works in mm by convention; convert once, here.
        mm = 1000.0
        plate = (
            cq.Workplane("XY")
            .workplane(offset=h * mm)
            .rect(2 * top_span * mm, 2 * top_span * mm)
            .extrude(pthk * mm)
            .edges("|Z")
            .fillet(0.25 * top_span * mm)
            .faces(">Z")
            .workplane()
            .hole(self.motor_dia * mm)
        )

        result = plate
        for cx, cy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            p0 = cq.Vector(cx * span * mm, cy * span * mm, 0)
            p1 = cq.Vector(cx * top_span * mm, cy * top_span * mm, h * mm)
            axis = p1.sub(p0)
            leg = (
                cq.Workplane("XY")
                .transformed(offset=p0)
                .circle(dia * mm / 2)
                .circle(max(dia / 2 - thk, 1e-4) * mm)
                .extrude(axis.Length)
            )
            # Tilt the extrusion onto the real leg axis.
            zax = cq.Vector(0, 0, 1)
            rot_axis = zax.cross(axis)
            if rot_axis.Length > 1e-9:
                ang = np.degrees(np.arccos(zax.dot(axis) / axis.Length))
                leg = leg.rotate(
                    (p0.x, p0.y, p0.z),
                    (p0.x + rot_axis.x, p0.y + rot_axis.y, p0.z + rot_axis.z),
                    ang,
                )
            foot = (
                cq.Workplane("XY")
                .transformed(offset=cq.Vector(p0.x, p0.y, 0))
                .rect(2.2 * dia * mm, 2.2 * dia * mm)
                .extrude(1.5 * thk * mm)
                .faces(">Z")
                .workplane()
                .hole(self.bolt_dia * mm)
            )
            result = result.union(leg).union(foot)
        return result

    def export_step(self, params: dict[str, float], path: str) -> str:
        import cadquery as cq  # noqa: F401  (validated in build)

        solid = self.build(params)
        cq.exporters.export(solid, path)
        return path
