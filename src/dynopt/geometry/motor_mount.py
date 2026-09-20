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
from ..dynamics.modal import ModeFamilySpec
from .base import DesignSpace, DesignVariable, PartSpec, material_hints

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

#: Exact mass-correction factor for the plate's beam idealisation.
#:
#: The platform is represented by four perimeter beams of width ``top_span``
#: and four spokes of width ``top_span/2``. Their swept volumes are
#:
#:     perimeter  4 * (top_span * t) * (2 * top_span)   =      8 * top_span^2 * t
#:     spokes     4 * (top_span/2 * t) * (sqrt(2)*top_span) = 2*sqrt(2) * top_span^2 * t
#:
#: against a true plate of ``(2*top_span)^2 * t = 4 * top_span^2 * t``. The
#: members overlap at the corners and hub, so the idealisation double-counts
#: material by ``(8 + 2*sqrt(2))/4 = 2.71x``. Both sides scale identically with
#: ``top_span`` and ``t``, so the correction is an exact constant, not a fit:
#:
#:     f = 4 / (8 + 2*sqrt(2)) = 0.369398...
#:
#: Applied to inertia only (see ``Section.mass_factor``); the plate's bending
#: stiffness representation is unchanged.
PLATE_MASS_FACTOR = 4.0 / (8.0 + 2.0 * np.sqrt(2.0))

#: Radial clearance of the plate edge beyond the leg outer surface [m].
PLATE_OVERHANG = 0.001

#: Smallest bore that is both manufacturable and meshable [m].
#:
#: The design box allows ``leg_thk`` up to 6 mm and ``leg_dia`` down to 8 mm, so
#: a wall thicker than the radius is reachable -- one sampled design asked for a
#: 5.18 mm wall on a 9.02 mm tube, i.e. a bore of -1.33 mm. Clamping the inner
#: radius to a hair above zero produced a sliver annulus that Gmsh rejected with
#: "invalid boundary mesh (overlapping facets)". Below this bore the leg simply
#: *is* solid, which is the physical answer and meshes cleanly.
MIN_BORE = 0.0005


def leg_bore_radius(dia: float, thk: float) -> float:
    """Inner radius of a leg, or 0.0 when the section is effectively solid.

    Used by the frame model and the CAD alike so the two tiers never disagree
    about whether a leg is hollow.
    """
    bore = 0.5 * float(dia) - float(thk)
    return bore if bore >= MIN_BORE else 0.0


def plate_geometry(params: dict[str, float], motor_dia: float = 0.028) -> dict:
    """Derived plate dimensions, shared by the frame model and the CAD.

    Both representations must agree or the tiers cannot be compared. Two rules
    here exist purely to keep the CAD valid across the *whole* design space --
    geometry that is only valid at the default design is useless to an
    optimiser:

    ``half_width``
        The plate overhangs the legs by ``PLATE_OVERHANG`` so each leg lands
        with its full footprint on solid material. Legs placed exactly on a
        plate corner overhang it by three quarters of their section, leaving
        slivers that mesh as "overlapping facets" or fail to fuse at all.

    ``hole_radius``
        Clamped so the motor bore can never reach the legs. A fixed 28 mm bore
        collides with them below about span = 55 mm, producing a self-
        intersecting solid.
    """
    span = float(params["span"])
    dia = float(params["leg_dia"])
    top_span = 0.55 * span
    half_width = top_span + 0.5 * dia + PLATE_OVERHANG
    max_bore = max(top_span - 0.5 * dia - 0.002, 0.003)
    return {
        "top_span": top_span,
        "half_width": half_width,
        "hole_radius": min(0.5 * motor_dia, max_bore),
    }

#: Effective-width multiplier on the plate's beam members (stiffness only).
#:
#: Correcting the mass alone leaves the idealisation ~21% stiff, because the
#: overlapping perimeter and spoke members double-count *stiffness* as well as
#: material. Rather than scale EI by an opaque constant, the calibrated quantity
#: is the members' effective width -- the classical idealisation parameter for
#: how much of a plate acts with a stiffener. Mass stays exact regardless,
#: since the mass factor is derived from the width (see ``plate_sections``).
#:
#: Calibrated against mesh-converged 3D CalculiX solves; see
#: ``examples/05_cross_solver_validation.py``.
PLATE_WIDTH_FACTOR = 1.0  # placeholder, calibrated below

#: Effective-width multiplier on the plate's *spoke* members (stiffness only).
#:
#: The spokes carry the payload into the plate, and four narrow strips in
#: bending reproduce only a fraction of a plate's two-dimensional stiffness.
#: Sized as the perimeter members were, they made the idealisation **11.4x too
#: soft at the plate centre** -- which is where the motor sits, so it showed up
#: as the whole of the beam tier's loaded error.
#:
#: The *form* comes from equating the four-spoke central stiffness to plate
#: theory, which is why this is an effective width and not a stiffness fudge::
#:
#:     k_spokes = E * w * t^3 / L^3,   L = a / sqrt(2)      (four spokes)
#:     k_plate  = c * D / a^2,         D = E t^3 / 12(1-nu^2)
#:     =>  w   proportional to a
#:
#: so the effective width scales with the platform span, exactly as the code
#: already assumed -- only the coefficient was wrong. Simply-supported plate
#: theory (c = 1/0.0116) gives 5.7; the real plate is carried on four discrete
#: leg tops rather than a continuous edge, which is softer, so the coefficient
#: is calibrated rather than taken from the closed form.
#:
#: Calibrated against 24 mesh-converged 3D designs, bare and loaded, fitted on
#: half and reported on the other half. Held-out first-frequency error:
#:
#:     ==========  ==========  ==========
#:     factor      bare        loaded
#:     ==========  ==========  ==========
#:     1.0 (old)   10.1%       40.8%
#:     2.5         7.5%        13.7%
#:     ==========  ==========  ==========
#:
#: Both improve, so this is not a trade. Only this one constant is calibrated:
#: PLATE_WIDTH_FACTOR stays 1.0, because fitting it too bought no held-out
#: accuracy and a second fitted constant has to earn its place.
#:
#: See ``docs/06_MULTIFIDELITY.md`` S6.6 and ``tests/test_plate_calibration.py``.
PLATE_SPOKE_FACTOR = 2.5

#: How this part's modes group onto rigid-body motions of the payload.
#:
#: ``sway_rock`` deliberately merges the two lateral translations with the two
#: rocking rotations. On a payload carried by compliant legs these are **not**
#: separable: they are inertially coupled through the payload's CG offset,
#: producing two mixed branches whose character swaps as the height/span ratio
#: changes. A mode signature of ``[0.66 ux, 0.66 ry]`` is a real mixed mode, and
#: deciding its label by argmax over two tied components is a coin flip that
#: injects ~50% noise into the surrogate target. One family, two branches.
MOTOR_MOUNT_FAMILIES = ModeFamilySpec(
    dofs={"bounce_z": (2,), "sway_rock": (0, 1, 3, 4), "torsion_z": (5,)},
    branches={"bounce_z": 1, "sway_rock": 2, "torsion_z": 1},
)

#: Target column names for this part, in order.
MODE_FAMILIES = MOTOR_MOUNT_FAMILIES.columns

#: Non-structural payload carried at the platform centre.
DEFAULT_MOTOR_MASS = 0.35  # kg
DEFAULT_MOTOR_INERTIA = (2.0e-4, 2.0e-4, 3.5e-4)  # kg.m^2 about x, y, z


def build_frame(
    params: dict[str, float],
    material: Material | str = "AL6061",
    motor_mass: float = DEFAULT_MOTOR_MASS,
    motor_inertia: tuple[float, float, float] = DEFAULT_MOTOR_INERTIA,
    n_seg: int = 4,
    plate_mass_factor: float = PLATE_MASS_FACTOR,
    plate_width_factor: float = PLATE_WIDTH_FACTOR,
    plate_spoke_factor: float = PLATE_SPOKE_FACTOR,
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

    bore = leg_bore_radius(dia, thk)
    leg_sec = (
        Section.tube(dia, 0.5 * dia - bore) if bore > 0.0 else Section.circle(dia)
    )
    # The plate is modelled as four edge beams of equivalent bending stiffness:
    # width = the tributary half-span, depth = the plate thickness. Their swept
    # volumes overlap, so inertia is corrected by PLATE_MASS_FACTOR while the
    # stiffness those widths imply is left alone.
    # Effective width sets stiffness; the mass factor is then derived from that
    # same width so the plate's true mass is reproduced exactly whatever the
    # calibration. Swept volume is alpha*(8 + 2*sqrt(2))*top_span^2*t against a
    # true plate of 4*top_span^2*t.
    alpha = max(plate_width_factor, 1e-6)
    plate_sec = Section.rectangle(max(alpha * top_span, 1e-4), pthk)
    # Spokes from the platform corners to the motor node.
    #
    # Their effective width is NOT the same as the perimeter's. Four narrow
    # strips in bending carry only a fraction of a plate's two-dimensional
    # stiffness, and the payload sits right where that matters -- see
    # PLATE_SPOKE_FACTOR.
    spoke_sec = Section.rectangle(
        max(0.5 * plate_spoke_factor * top_span, 1e-4), pthk
    )

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

    # Correct the plate members' inertia to the true plate mass.  Computed from
    # the actual swept volumes rather than a closed-form constant, so it stays
    # exact if the idealisation's topology or member widths ever change.
    n_leg_elems = 4 * n_seg
    plate_swept = sum(
        sections[e].A * np.linalg.norm(np.asarray(nodes[j]) - np.asarray(nodes[i]))
        for e, (i, j) in enumerate(elements)
        if e >= n_leg_elems
    )
    geom = plate_geometry(params)
    true_plate = (2.0 * geom["half_width"]) ** 2 * pthk - np.pi * geom["hole_radius"] ** 2 * pthk
    mf = (true_plate / plate_swept) if plate_swept > 0 else 1.0
    for e in range(n_leg_elems, len(sections)):
        sec = sections[e]
        sections[e] = Section(
            A=sec.A, Iy=sec.Iy, Iz=sec.Iz, J=sec.J, Ip=sec.Ip,
            kappa=sec.kappa, mass_factor=mf,
        )

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


class MotorMount(PartSpec):
    """The benchmark part, as a self-contained :class:`PartSpec`.

    Supplies both tiers: a frame idealisation for fast DOE work and a CadQuery
    solid for 3D verification. Everything the pipeline knows about this part
    lives in this module -- the core carries no reference to it.

    CadQuery is imported lazily so the package stays usable without the (large)
    OCP dependency; the frame tier needs only numpy and scipy.
    """

    name = "motor_mount"
    # Measured 1.175 at the default design (beam 372.6 Hz, 3D 437.9 Hz), down
    # from 1.735 before the plate spokes were calibrated -- see
    # PLATE_SPOKE_FACTOR. Tightened to match, because a bound loose enough to
    # admit the old idealisation would no longer catch a modelling bug.
    max_tier_ratio = 1.35

    def __init__(
        self,
        motor_dia: float = 0.028,
        bolt_dia: float = 0.004,
        material: str = "AL6061",
        motor_mass: float = DEFAULT_MOTOR_MASS,
        motor_inertia: tuple[float, float, float] = DEFAULT_MOTOR_INERTIA,
    ):
        self.motor_dia = motor_dia
        self.bolt_dia = bolt_dia
        self.material = material
        self.motor_mass = motor_mass
        self.motor_inertia = motor_inertia

    @property
    def design_space(self) -> DesignSpace:
        return MOTOR_MOUNT_SPACE

    @property
    def mode_families(self) -> ModeFamilySpec:
        return MOTOR_MOUNT_FAMILIES

    def build_frame(self, params: dict[str, float]) -> FrameModel:
        return build_frame(
            params,
            material=self.material,
            motor_mass=self.motor_mass,
            motor_inertia=self.motor_inertia,
        )

    def solver_hints(self, params: dict[str, float]) -> dict:
        """Clamp the named mounting plane, and smear the motor over its flange.

        ``z_plane`` rather than "lowest face": design variables move geometry
        around, and a lowest-face selector will happily clamp four isolated leg
        tips instead of the bolt-down faces, then solve and report nonsense.
        """
        geom = plate_geometry(params, self.motor_dia)
        bore = leg_bore_radius(params["leg_dia"], params["leg_thk"])
        thin = [float(params["plate_thk"]), float(params["leg_thk"])]
        if bore > 0.0:
            thin.append(2.0 * bore)
        return {
            "fix_selector": "z_plane",
            "fix_plane": 0.0,
            "fix_tol": 1e-3,
            # mm / tonne / s, matching the mm geometry CadQuery exports
            "payload_mass": self.motor_mass * 1e-3,
            "payload_radius": max(geom["hole_radius"] * 1000.0 * 1.6, 8.0),
            "min_feature": min(thin) * 1000.0,
            **material_hints(self.material),
        }

    def build_cad(self, params: dict[str, float]):
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
        geom = plate_geometry(params, self.motor_dia)
        top_span = geom["top_span"]
        half_w = geom["half_width"]
        bore = geom["hole_radius"]

        # CadQuery works in mm by convention; convert once, here.
        mm = 1000.0
        # No corner fillet.  Its radius scaled with the plate, so on larger
        # designs it removed exactly the material the legs land on and the part
        # fell back into disjoint volumes.  It was cosmetic; robustness across
        # the design space is not.
        plate = (
            cq.Workplane("XY")
            .workplane(offset=h * mm)
            .rect(2 * half_w * mm, 2 * half_w * mm)
            .extrude(pthk * mm)
            .faces(">Z")
            .workplane()
            .hole(2 * bore * mm)
        )

        result = plate
        for cx, cy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            p0 = cq.Vector(cx * span * mm, cy * span * mm, 0)
            p1 = cq.Vector(cx * top_span * mm, cy * top_span * mm, h * mm)
            axis = p1.sub(p0)
            # Run the leg THROUGH the plate rather than stopping at its
            # underside.  A leg that merely meets the plate face produces no
            # boolean intersection in OpenCASCADE, so `union` silently returns a
            # compound of disjoint solids: the STEP then carries five separate
            # volumes, the mesher meshes each independently, and the modal solve
            # reports six rigid-body modes because the platform is mechanically
            # attached to nothing.  Verified: embed = 0 gives 5 volumes,
            # embed >= plate_thk gives 1.
            bore = leg_bore_radius(dia, thk)
            leg_wp = cq.Workplane("XY").transformed(offset=p0).circle(dia * mm / 2)
            if bore > 0.0:
                leg_wp = leg_wp.circle(bore * mm)
            leg = leg_wp.extrude(axis.Length + pthk * mm)
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

        # Trim everything below the mounting plane.  Each leg is extruded
        # vertically and then rotated onto its true axis, which swings the
        # tilted end face below z = 0 -- on the default design by 4.2 mm.  Left
        # in, the part has no flat bolt-down face at all: a geometric "lowest
        # face" selector then clamps four isolated leg tips instead of the feet,
        # and the modal result is meaningless.  Caught by cross-checking the
        # meshed solid against the beam idealisation.
        below = (
            cq.Workplane("XY")
            .box(16 * span * mm, 16 * span * mm, 20 * h * mm, centered=(True, True, False))
            .translate((0, 0, -20 * h * mm))
        )
        # Trim the legs flush with the top of the plate.  They are deliberately
        # extruded past it (see above) so the boolean union actually fuses;
        # without this second cut they protrude, which both looks wrong and
        # breaks any "select the top face" query -- the highest nodes would be
        # four leg tips rather than the mounting surface.
        above = (
            cq.Workplane("XY")
            .box(16 * span * mm, 16 * span * mm, 20 * h * mm, centered=(True, True, False))
            .translate((0, 0, (h + pthk) * mm))
        )
        return result.cut(below).cut(above)

    #: Back-compatible alias: earlier code and the examples call ``build()``.
    build = build_cad

    def export_step(self, params: dict[str, float], path: str) -> str:
        import cadquery as cq  # noqa: F401  (validated in build_cad)

        cq.exporters.export(self.build_cad(params), path)
        return path


#: The part, ready to hand to a solver.
MOTOR_MOUNT = MotorMount()

#: Back-compatible alias from when this class was CAD-only.
MotorMountCAD = MotorMount


#: Leg slenderness (L/d) below which Euler-Bernoulli/Timoshenko beam theory
#: stops representing a member. Measured against 3D solid solves, designs below
#: this ran -16.8% mean error; above it, -1.4% mean.
MIN_SLENDERNESS = 4.5

#: Plate aspect ratio (width / thickness) above which the edge-beam-plus-spoke
#: idealisation loses the plate's two-dimensional action. Error correlates with
#: this more strongly (r = -0.78) than with leg slenderness.
MAX_PLATE_ASPECT = 35.0


def tier0_validity(params: dict[str, float]) -> dict:
    r"""Report whether a design sits inside Tier 0's validated envelope.

    A beam idealisation is not a general-purpose solver, and it will return a
    confident number for a design it cannot represent. Calibration against
    mesh-converged 3D solves showed the error is **design-dependent with both
    signs** -- from -27.5% to +4.8% -- so no scalar correction fixes it:

    ======================  ==========  ===============
    Envelope                 n          max abs error
    ======================  ==========  ===============
    ``L/d >= 4.5``           5           11.1%
    ``L/d <  4.5``           2           27.5%
    ======================  ==========  ===============

    Returns the two governing ratios and an ``ok`` flag. Use it to gate which
    designs may be quoted from Tier 0 alone, and which must go to Tier 1. It is
    advisory, not enforced: the optimiser is free to explore outside the
    envelope, but anything it returns from out there needs a 3D sign-off.
    """
    span = float(params["span"])
    dia = float(params["leg_dia"])
    top_span = 0.55 * span
    dx = span - top_span
    length = float(np.sqrt(2.0 * dx**2 + float(params["height"]) ** 2))
    slenderness = length / dia
    aspect = (2.0 * top_span) / float(params["plate_thk"])
    return {
        "slenderness": slenderness,
        "plate_aspect": aspect,
        "ok": slenderness >= MIN_SLENDERNESS and aspect <= MAX_PLATE_ASPECT,
        "reasons": [
            *( [f"leg L/d = {slenderness:.2f} < {MIN_SLENDERNESS}"]
               if slenderness < MIN_SLENDERNESS else [] ),
            *( [f"plate aspect = {aspect:.1f} > {MAX_PLATE_ASPECT}"]
               if aspect > MAX_PLATE_ASPECT else [] ),
        ],
    }
