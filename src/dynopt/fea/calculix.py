r"""Open-source FEA backend: STEP -> Gmsh -> CalculiX -> ModalResult.

Pipeline
--------
1. **Gmsh** reads the STEP directly through its OpenCASCADE kernel
   (``gmsh.model.occ.importShapes``), so there is no intermediate format and no
   geometry healing step.
2. Mesh with second-order tetrahedra (C3D10).  *Never use linear tets (C3D4) for
   modal analysis* -- they are numerically over-stiff and typically report
   natural frequencies 10-25% high.  This single choice causes more bad modal
   results than anything else in FEA automation.
3. Write a CalculiX ``.inp`` deck with a ``*FREQUENCY`` step.
4. Run ``ccx`` and parse eigenvalues from ``.dat`` and mode shapes from ``.frd``.

Install
-------
``pip install 'dynopt[oss]'`` for gmsh/meshio; CalculiX itself is a system
binary: ``apt install calculix-ccx`` (Debian/Ubuntu), ``brew install calculix``
(macOS), or the ``ccx`` executable from calculix.de on Windows.

Licensing note
--------------
CalculiX is GPL and Gmsh is GPL -- both are free for commercial use, which
matters if you intend to sell this.  Bundling *Ansys* results into a commercial
product is a very different licensing conversation, and an academic Ansys
licence explicitly forbids commercial use.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp

from .base import ModalResult, ModalSolver


_GMSH_STARTED = False


def _ensure_gmsh():
    """Initialise Gmsh once per process and keep it alive.

    Gmsh does not tolerate repeated ``initialize()``/``finalize()`` cycles: after
    enough of them it aborts the process inside ``mesh``, with no Python
    exception to catch. It surfaced first in the integration tests, but it is a
    production bug -- a Tier 1 DOE meshes hundreds of designs in one process and
    would die partway through a long run.

    One session for the process lifetime, the model cleared between designs, and
    finalisation left to ``atexit``.
    """
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("gmsh is required: pip install 'dynopt[oss]'") from exc

    global _GMSH_STARTED
    try:
        live = bool(gmsh.isInitialized())
    except AttributeError:  # gmsh < 4.10
        live = _GMSH_STARTED
    if not live:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        if not _GMSH_STARTED:
            atexit.register(_shutdown_gmsh)
        _GMSH_STARTED = True
    return gmsh


def _shutdown_gmsh():  # pragma: no cover - runs at interpreter exit
    global _GMSH_STARTED
    if not _GMSH_STARTED:
        return
    _GMSH_STARTED = False
    try:
        import gmsh

        if gmsh.isInitialized():
            gmsh.finalize()
    except Exception:
        pass


def _reset_gmsh():
    """Tear the Gmsh library down and bring it back up.

    Only ever called after a failure. ``gmsh.clear()`` is not enough to recover
    from one: once ``mesh.generate`` has thrown, the session keeps accepting
    calls but every later ``generate`` returns an *empty* mesh, and the error
    surfaces hundreds of lines downstream as an empty-array reduction. One bad
    design in a DOE therefore poisons every design after it.

    Repeated init/finalise cycles are what ``_ensure_gmsh`` exists to avoid, so
    this stays bounded to the failure path: a DOE of hundreds of designs still
    runs on one session unless something actually breaks.
    """
    global _GMSH_STARTED
    try:
        import gmsh

        if gmsh.isInitialized():
            gmsh.finalize()
    except Exception:  # pragma: no cover - best effort teardown
        pass
    _GMSH_STARTED = False


@contextmanager
def gmsh_model(name: str = "part"):
    """Scoped Gmsh model on the shared session.

    Clears the previous model rather than tearing down the library, *except*
    when the body raises -- see :func:`_reset_gmsh` for why a failure has to
    take the whole session with it. Use this everywhere instead of
    ``gmsh.initialize()`` / ``gmsh.finalize()``.
    """
    gmsh = _ensure_gmsh()
    gmsh.clear()
    gmsh.model.add(name)
    try:
        yield gmsh
    except BaseException:
        _reset_gmsh()
        raise
    else:
        try:
            gmsh.clear()
        except Exception:
            pass


@dataclass
class CalculiXConfig:
    """Solver and meshing settings.

    Units are **mm / tonne / s**, because that is what CadQuery exports and a
    deck must agree with its geometry. The old SI defaults (68.9e9, 2700) were
    a trap: paired with millimetre geometry they are wrong by nine orders of
    magnitude in density, and the solve still converges. A part attached to the
    solver supplies its own material through ``solver_hints``, so these
    defaults only apply to a bare ``ParametricModel``.
    """

    E: float = 68900.0                # AL6061 [MPa]
    nu: float = 0.33
    rho: float = 2.7e-9               # AL6061 [tonne/mm^3]
    n_modes: int = 10
    mesh_size: float = 3.0            # target element size [mm]
    mesh_size_min: float = 0.8
    order: int = 2                    # 2 => C3D10; do not lower this
    fix_selector: str = "z_min"       # "z_min" | "z_plane" | "all"
    fix_plane: float = 0.0            # used when fix_selector == "z_plane"
    fix_tol: float = 1e-3             # ABSOLUTE, in model units (mm)
    # Which face the payload sits on: "<x|y|z>_<min|max>". The radius is
    # measured in the plane perpendicular to that axis. A part declares it,
    # because where the motor bolts on is a property of the geometry.
    payload_selector: str = "z_max"
    solid_type: str = "C3D10"         # element type to keep from the gmsh mesh
    # Gmsh's high-order optimiser pulls mid-side nodes onto the true CAD
    # surface, which is worth a little accuracy on curved features. It is also
    # capable of throwing an uncaught C++ exception that calls terminate() and
    # kills the process -- no Python exception, nothing to catch, a long DOE
    # simply dies. Measured on this benchmark: level 1 aborts at 3.0 mm
    # elements, level 2 aborts at 5.0 mm, and which combination triggers it is
    # not predictable. Level 0 is the only reliably safe setting and is the
    # default. Raise it only for a one-off run you are watching.
    high_order_optimize: int = 0
    # Smallest wall or bore in this design [model units], supplied per design by
    # ``PartSpec.solver_hints``. 0 means "unknown".
    #
    # It does NOT cap the global element size. That was tried and measured: a
    # global cap refined every design in both boxes, ten of them finer than the
    # *fine* tier, and one coarse-tier mesh reached 159k nodes -- "coarse"
    # became more expensive than "fine". Its only job now is to stop
    # ``mesh_size_min`` from clamping the curvature refinement below.
    min_feature: float = 0.0
    # Elements around a full circle of local curvature. Gmsh then refines bores
    # and tube walls -- the features an element actually has to wrap -- while
    # leaving flat regions at ``mesh_size``. 0 disables it.
    #
    # Measured over 24 Sobol designs of each part at a 6 mm target, against a
    # fixed element size: 48/48 designs meshed instead of 39/48, for 1.7x the
    # nodes on the mount and 1.8x on the arm. Every design a fixed size lost
    # failed the same way -- an element larger than a bore it had to wrap.
    mesh_from_curvature: int = 12
    # Straight-sided quadratic tets: mid-side nodes at true edge midpoints
    # instead of projected onto the CAD surface.
    #
    # Curved second-order tets on a tight bore routinely invert, and CalculiX
    # rejects them with "nonpositive jacobian determinant" -- which is why the
    # high-order optimiser exists. But that optimiser can abort the process
    # outright, so neither curved-and-optimised nor curved-and-unoptimised is
    # safe unattended. Straight-sided elements are quadratic in displacement,
    # which is what avoids the over-stiff linear tet, and cannot invert. The
    # cost is a slightly coarser geometric fit on curved surfaces, recovered by
    # refining the mesh.
    second_order_linear: bool = True
    ccx_path: str = "ccx"
    export_matrices: bool = False     # *MATRIX OUTPUT (version dependent)
    keep_files: bool = False
    # Non-structural payload, smeared over the nodes of the top face within
    # `payload_radius` of the part axis -- a motor bolted to its flange.  Put on
    # a single node it would instead excite a spurious local mode.
    # Model mass units (tonne for mm/t/s). ``None`` means "ask the part"; a
    # number -- including ``0.0`` -- is the caller's own choice and is never
    # overridden.
    #
    # This field cannot use the "differs from the default" rule the other
    # hintable fields use, because zero is a *meaningful* value here: a bare
    # structure run asks for exactly 0.0 payload, which is indistinguishable
    # from never having asked. Example 05 did exactly that, and its "bare
    # structure" case silently came back carrying the 350 g motor -- printing
    # the same frequency as its loaded case, with nothing to say the two runs
    # were identical.
    payload_mass: float | None = None
    payload_radius: float = 0.0       # model length units (mm)


def mesh_step(
    step_path: str, out_inp: str, cfg: CalculiXConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a STEP file with Gmsh and write an Abaqus/CalculiX ``.inp``.

    Returns ``(node_coords, node_tags)``.  Gmsh writes Abaqus format natively,
    which CalculiX reads with only minor keyword additions.
    """
    with gmsh_model("part") as gmsh:
        gmsh.model.occ.importShapes(step_path)

        # A STEP file routinely contains several solids that merely *touch*
        # rather than overlap -- CAD booleans leave tangent contacts unfused, so
        # what looks like one part arrives as a compound.  Meshed as-is, each
        # solid gets its own nodes on the shared face and the assembly is
        # mechanically disconnected: the modal solve then returns six
        # rigid-body modes for whatever floats free of the constraints, with no
        # warning from either mesher or solver.
        #
        # `fragment` imprints the volumes on one another so they share interface
        # faces and the mesh comes out conformal (a bonded joint, which is the
        # right default for a welded or monolithic part).  This is essential for
        # arbitrary customer geometry, not just this benchmark.
        vols = gmsh.model.occ.getEntities(3)
        if len(vols) > 1:
            gmsh.model.occ.fragment(vols[:1], vols[1:])
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        n_vol = len(gmsh.model.getEntities(3))

        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", float(cfg.mesh_from_curvature)
        )
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cfg.mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cfg.mesh_size_min)
        gmsh.option.setNumber("Mesh.ElementOrder", cfg.order)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", cfg.high_order_optimize)
        gmsh.option.setNumber(
            "Mesh.SecondOrderLinear", 1 if cfg.second_order_linear else 0
        )

        gmsh.model.mesh.generate(3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)

        # Gmsh can return successfully having meshed nothing. Catch it here,
        # where the cause is still visible, rather than letting an empty node
        # array travel into node selection and surface as a numpy error about a
        # zero-size reduction.
        n_solid = 0
        for _, vol in gmsh.model.getEntities(3):
            for block in gmsh.model.mesh.getElements(3, vol)[1]:
                n_solid += len(block)
        if coords.shape[0] == 0 or n_solid == 0:
            raise RuntimeError(
                f"Gmsh produced an empty mesh for {step_path} "
                f"({coords.shape[0]} nodes, {n_solid} solid elements) at "
                f"mesh_size={cfg.mesh_size:g}/{cfg.mesh_size_min:g}"
            )

        gmsh.write(out_inp)

        # Guard the failure above: after fragmenting, a conformal mesh must not
        # contain coincident duplicate nodes.  Duplicates here mean the solids
        # were never joined and the result would be silently wrong.
        if n_vol > 1:
            uniq = np.unique(np.round(coords, 6), axis=0)
            if uniq.shape[0] < coords.shape[0] - 1:
                raise RuntimeError(
                    f"mesh has {coords.shape[0] - uniq.shape[0]} coincident "
                    f"duplicate nodes across {n_vol} volumes: the solids are not "
                    "bonded and the model would be mechanically disconnected"
                )
        return coords, np.asarray(tags, dtype=int)


def count_volumes(step_path: str) -> int:
    """Number of distinct solid volumes in a STEP file.

    More than one means the CAD booleans did not fuse and the part will mesh as
    disconnected pieces.
    """
    with gmsh_model("volcheck") as gmsh:
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        return len(gmsh.model.getEntities(3))


def _select_payload_nodes(coords: np.ndarray, tags: np.ndarray, cfg: CalculiXConfig):
    """Nodes carrying the smeared payload: a disc on one face of the part.

    ``payload_selector`` names the face (``"z_max"`` for a mount's platform,
    ``"x_max"`` for an arm's tip boss) and ``payload_radius`` is measured in the
    plane perpendicular to that axis.

    This was hardcoded to ``z`` with the radius measured from the z axis, which
    is a motor mount's shape. On the cantilever arm -- which runs along x -- it
    selected nodes near the *root* collar and smeared the motor there. The arm
    solved happily and reported 338 Hz where a clamped tube with an 85 g tip
    mass gives 116 Hz: the payload was attached where it does no work. Nothing
    in the pipeline objected, and the 3x error reads exactly like a
    beam-idealisation discrepancy waiting to be "corrected".
    """
    axes = {"x": 0, "y": 1, "z": 2}
    try:
        axis_name, kind = cfg.payload_selector.split("_", 1)
        axis = axes[axis_name]
    except (ValueError, KeyError):
        raise ValueError(
            f"unknown payload_selector {cfg.payload_selector!r}; expected "
            "'<x|y|z>_<min|max>'"
        ) from None

    v = coords[:, axis]
    if kind == "max":
        on_face = v >= v.max() - cfg.fix_tol
    elif kind == "min":
        on_face = v <= v.min() + cfg.fix_tol
    else:
        raise ValueError(
            f"unknown payload_selector {cfg.payload_selector!r}; expected "
            "'<x|y|z>_<min|max>'"
        )

    perp = [i for i in range(3) if i != axis]
    rad = np.hypot(coords[:, perp[0]], coords[:, perp[1]])
    sel = on_face & (rad <= cfg.payload_radius)
    if not sel.any():
        reach = rad[on_face].max() if on_face.any() else float("nan")
        raise ValueError(
            f"payload_radius={cfg.payload_radius:g} selected no nodes on the "
            f"{cfg.payload_selector} face ({int(on_face.sum())} nodes there, "
            f"largest radius {reach:.3g})"
        )
    return tags[sel]


def _select_fixed_nodes(coords: np.ndarray, tags: np.ndarray, cfg: CalculiXConfig):
    """Pick clamped nodes geometrically (the bolt-down face).

    Selectors are ``"<axis>_plane"`` or ``"<axis>_min"`` / ``"<axis>_max"`` for
    any of ``x``, ``y``, ``z``, plus ``"all"``. A part declares which it needs
    through :meth:`PartSpec.solver_hints`, because which face is the mounting
    face is a property of the geometry: a mount clamps ``z_plane`` at its base,
    a cantilever arm clamps ``x_plane`` at its root.

    Prefer ``"<axis>_plane"`` over ``"<axis>_min"``. A named plane stays correct
    when a design variable moves geometry around; "whatever is lowest" does not,
    and when it picks the wrong face the solve still succeeds and reports
    confident nonsense.

    ``fix_tol`` is **absolute**, in model units, not a fraction of the bounding
    box. A relative tolerance looks safer but silently rescales with the part.
    """
    axes = {"x": 0, "y": 1, "z": 2}
    selector = cfg.fix_selector

    if selector == "all":
        sel = np.ones(coords.shape[0], dtype=bool)
        axis_name, v = "z", coords[:, 2]
    else:
        try:
            axis_name, kind = selector.split("_", 1)
            v = coords[:, axes[axis_name]]
        except (ValueError, KeyError):
            raise ValueError(
                f"unknown fix_selector {selector!r}; expected 'all' or "
                "'<x|y|z>_<plane|min|max>'"
            ) from None
        if kind == "plane":
            sel = np.abs(v - cfg.fix_plane) <= cfg.fix_tol
        elif kind == "min":
            sel = v <= v.min() + cfg.fix_tol
        elif kind == "max":
            sel = v >= v.max() - cfg.fix_tol
        else:
            raise ValueError(
                f"unknown fix_selector {selector!r}; expected 'all' or "
                "'<x|y|z>_<plane|min|max>'"
            )

    if not sel.any():
        raise ValueError(
            f"fix_selector {selector!r} matched no nodes ({axis_name} range "
            f"{v.min():.4g}..{v.max():.4g}, plane {cfg.fix_plane:g}, "
            f"tol {cfg.fix_tol:g})"
        )
    return tags[sel]


def _split_mesh_inp(path: str, solid_type: str) -> tuple[list[str], list[str]]:
    """Split a Gmsh-written .inp into its node block and *solid* elements only.

    Gmsh meshes every entity it knows about, so a 3D mesh file also contains the
    1D elements of the curves (``T3D3``) and the 2D elements of the surfaces
    (``CPS6``).  Handing those to CalculiX is fatal -- they need their own
    sections, and any that survive contribute spurious low-frequency modes.

    Gmsh also names its element sets after geometric entities (``Volume1``,
    ``Surface3``); it never emits the Abaqus-conventional ``EALL``.  Referencing
    ``EALL`` in a ``*SOLID SECTION`` aborts the run with
    "element set EALL has not yet been defined".

    Returns ``(node_lines, element_lines)``, the element lines renumbered into a
    single set this module controls.
    """
    nodes: list[str] = []
    elems: list[str] = []
    mode = None
    keep = False

    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("*"):
                key = stripped.upper()
                if key.startswith("*NODE"):
                    mode, keep = "node", True
                elif key.startswith("*ELEMENT"):
                    mode = "elem"
                    keep = f"TYPE={solid_type.upper()}" in key.replace(" ", "")
                else:
                    mode, keep = None, False
                continue
            if not stripped or not keep:
                continue
            (nodes if mode == "node" else elems).append(stripped)
    return nodes, elems


def write_modal_deck(
    inp_mesh: str,
    out_inp: str,
    fixed_tags: np.ndarray,
    cfg: CalculiXConfig,
    payload_tags: np.ndarray | None = None,
) -> str:
    """Write a self-contained CalculiX ``*FREQUENCY`` deck.

    The mesh is rebuilt from the Gmsh file rather than included verbatim, so
    only solid elements reach the solver and the element set the material is
    assigned to is one we defined ourselves.
    """
    node_lines, elem_lines = _split_mesh_inp(inp_mesh, cfg.solid_type)
    if not elem_lines:
        raise ValueError(
            f"no {cfg.solid_type} elements in {inp_mesh}; "
            "check Mesh.ElementOrder and that the STEP contains a solid"
        )

    lines = ["*HEADING", "dynopt modal analysis", "*NODE"]
    lines += node_lines
    lines.append(f"*ELEMENT, TYPE={cfg.solid_type}, ELSET=ESOLID")
    lines += elem_lines

    mass = cfg.payload_mass or 0.0
    if mass > 0.0 and payload_tags is not None and payload_tags.size:
        share = mass / payload_tags.size
        start = max(int(e.split(",")[0]) for e in elem_lines) + 1
        lines.append("*ELEMENT, TYPE=MASS, ELSET=EPAYLOAD")
        for k, tag in enumerate(payload_tags):
            lines.append(f"{start + k}, {tag}")
        lines += ["*MASS, ELSET=EPAYLOAD", f"{share:.8e}"]

    lines.append("*NSET, NSET=FIXED")
    for i in range(0, fixed_tags.size, 8):
        lines.append(", ".join(str(t) for t in fixed_tags[i : i + 8]) + ",")

    lines += [
        "*MATERIAL, NAME=PARTMAT",
        "*ELASTIC",
        f"{cfg.E:.6e}, {cfg.nu:.4f}",
        "*DENSITY",
        f"{cfg.rho:.6e}",
        "*SOLID SECTION, ELSET=ESOLID, MATERIAL=PARTMAT",
        "*BOUNDARY",
        "FIXED, 1, 3",
        "*STEP",
        "*FREQUENCY, STORAGE=YES",
        f"{cfg.n_modes}",
        "*NODE FILE",
        "U",
    ]
    if cfg.export_matrices:
        lines.append("*MATRIX OUTPUT, STIFFNESS, MASS")
    lines += ["*END STEP", ""]

    with open(out_inp, "w") as fh:
        fh.write("\n".join(lines))
    return out_inp


def parse_dat_frequencies(dat_path: str) -> np.ndarray:
    """Pull natural frequencies (Hz) from a CalculiX ``.dat`` file.

    CalculiX prints an ``E I G E N V A L U E   O U T P U T`` block whose
    columns are mode number, eigenvalue, angular frequency (rad/time) and
    frequency (cycles/time).  We take cycles/time and fall back to
    :math:`\\sqrt{\\lambda}/2\\pi` if the column layout differs by version.
    """
    with open(dat_path) as fh:
        text = fh.read()

    m = re.search(r"E\s*I\s*G\s*E\s*N\s*V\s*A\s*L\s*U\s*E", text)
    if not m:
        raise ValueError(f"no eigenvalue block found in {dat_path}")

    freqs = []
    for line in text[m.end():].splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", line)
        if len(nums) >= 4:
            vals = [float(n.replace("d", "e").replace("D", "E")) for n in nums]
            freqs.append(vals[3])       # cycles/time column
        elif len(nums) >= 2 and freqs:
            break                        # end of the block
    if not freqs:
        raise ValueError(f"eigenvalue block in {dat_path} had no parsable rows")
    return np.array(freqs, dtype=float)


def parse_frd_modes(frd_path: str, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Read displacement mode shapes from a CalculiX ``.frd`` result file.

    Returns ``(node_tags, modes)`` with ``modes`` shaped ``(3*n_nodes, n_modes)``
    in ``[ux, uy, uz]`` per-node order.
    """
    blocks: list[dict[int, tuple[float, float, float]]] = []
    current: dict[int, tuple[float, float, float]] | None = None

    with open(frd_path) as fh:
        for line in fh:
            if line.startswith(" -4") and "DISP" in line:
                current = {}
                blocks.append(current)
            elif line.startswith(" -3"):
                current = None
            elif line.startswith(" -1") and current is not None:
                try:
                    node = int(line[3:13])
                    vals = [float(line[13 + 12 * k : 25 + 12 * k]) for k in range(3)]
                    current[node] = tuple(vals)
                except ValueError:
                    continue

    blocks = [b for b in blocks if b][:n_modes]
    if not blocks:
        raise ValueError(f"no DISP blocks parsed from {frd_path}")

    tags = np.array(sorted(blocks[0]), dtype=int)
    modes = np.zeros((3 * tags.size, len(blocks)))
    index = {t: i for i, t in enumerate(tags)}
    for j, b in enumerate(blocks):
        for t, (ux, uy, uz) in b.items():
            i = index[t]
            modes[3 * i : 3 * i + 3, j] = (ux, uy, uz)
    return tags, modes


class CalculiXSolver(ModalSolver):
    """Full CAD-to-modal backend using only GPL tooling.

    ``cad_model`` must implement :meth:`ParametricModel.export_step`.
    """

    name = "gmsh-calculix"

    def __init__(self, cad_model, cfg: CalculiXConfig | None = None, workdir: str | None = None):
        self.cad = cad_model
        self.cfg = cfg or CalculiXConfig()
        self.workdir = workdir
        #: True when ``cad_model`` is a PartSpec that can supply BC hints.
        self.part = cad_model if hasattr(cad_model, "solver_hints") else None
        #: Fields the caller pinned, decided once, at construction.
        #:
        #: "Differs from the default" cannot be re-derived later: the first
        #: solve writes hint values into ``cfg``, and from then on every
        #: design-dependent hint looks caller-set and stops updating. One solver
        #: driving a DOE would give every design the *first* design's payload
        #: radius and mesh cap, silently.
        defaults = CalculiXConfig()
        self._pinned = {
            f
            for f in vars(defaults)
            if getattr(self.cfg, f, None) != getattr(defaults, f)
        }

    def _apply_part_hints(self, params: dict[str, float]) -> None:
        """Refresh the config fields the part owns, for *this* design.

        Boundary conditions belong to the geometry, not the solver: a selector
        like "clamp the lowest face" silently picks the wrong face once a design
        variable moves things around, and the run succeeds anyway. Fields the
        caller pinned at construction are never overridden.
        """
        if self.part is None:
            return
        hints = self.part.solver_hints(params)
        if not hints:
            return
        for key, value in hints.items():
            if hasattr(self.cfg, key) and key not in self._pinned:
                setattr(self.cfg, key, value)

    def _mesh_sizes(self) -> tuple[float, float]:
        """Element sizes for this design: the target, and a floor that respects it.

        A fixed element size is not portable across a design space. The coarse
        tier asks for 6 mm elements; one sampled arm has a 5.5 mm bore through
        an 11 mm tube, and a 6 mm element cannot wrap that annulus -- Gmsh
        rejects the surface mesh as overlapping facets and the design is simply
        lost from the DOE.

        Curvature-driven refinement (``mesh_from_curvature``) handles that, but
        only if ``mesh_size_min`` leaves it room: a minimum size larger than the
        feature clamps the refinement away again and the design fails anyway.
        So the part's smallest feature lowers the *floor*, never the target.

        This only ever refines: a caller asking for a fine mesh still gets it.
        """
        h, hmin = self.cfg.mesh_size, self.cfg.mesh_size_min
        feat = float(self.cfg.min_feature)
        if feat > 0.0 and self.cfg.mesh_from_curvature > 0:
            hmin = min(hmin, feat / 3.0)
        return h, min(hmin, h)

    def solve(self, params: dict[str, float], n_modes: int = 10) -> ModalResult:
        self._apply_part_hints(params)
        cfg = self.cfg
        cfg.n_modes = max(n_modes, cfg.n_modes)

        if shutil.which(cfg.ccx_path) is None:
            raise RuntimeError(
                f"CalculiX executable {cfg.ccx_path!r} not found on PATH. "
                "Install it (apt install calculix-ccx) or set cfg.ccx_path."
            )

        tmp = self.workdir or tempfile.mkdtemp(prefix="dynopt_ccx_")
        os.makedirs(tmp, exist_ok=True)
        step = os.path.join(tmp, "part.step")
        mesh_inp = os.path.join(tmp, "mesh.inp")
        job = os.path.join(tmp, "job")

        try:
            self.cad.export_step(params, step)
            h, hmin = self._mesh_sizes()
            mesh_cfg = replace(cfg, mesh_size=h, mesh_size_min=hmin)
            coords, tags = mesh_step(step, mesh_inp, mesh_cfg)
            fixed = _select_fixed_nodes(coords, tags, cfg)

            payload = None
            if (cfg.payload_mass or 0.0) > 0.0:
                payload = _select_payload_nodes(coords, tags, cfg)
            write_modal_deck(mesh_inp, job + ".inp", fixed, cfg, payload_tags=payload)

            proc = subprocess.run(
                [cfg.ccx_path, "job"], cwd=tmp, capture_output=True, text=True, timeout=3600
            )
            # CalculiX creates job.dat and then aborts on a bad deck, so an
            # existence check is not enough -- an empty file means failure.
            if not os.path.exists(job + ".dat") or os.path.getsize(job + ".dat") == 0:
                raise RuntimeError(
                    "ccx produced no eigenvalue output (deck rejected or solve "
                    f"aborted).\nstdout:\n{proc.stdout[-3000:]}\n"
                    f"stderr:\n{proc.stderr[-2000:]}"
                )

            freqs = parse_dat_frequencies(job + ".dat")[:n_modes]
            modes = None
            if os.path.exists(job + ".frd"):
                try:
                    _, modes = parse_frd_modes(job + ".frd", n_modes)
                except Exception:
                    modes = None  # frequencies are still usable without shapes

            return ModalResult(
                frequencies=freqs,
                modes=modes,
                mass=float("nan"),
                meta={
                    "backend": self.name,
                    "params": dict(params),
                    **({"part": self.part} if self.part is not None else {}),
                    "n_nodes": int(coords.shape[0]),
                    "mesh_size": h,
                    "mesh_size_min": hmin,
                    "step_file": step,
                    "workdir": tmp,
                },
            )
        finally:
            if not cfg.keep_files and self.workdir is None:
                shutil.rmtree(tmp, ignore_errors=True)


def read_matrix_market_like(path: str, n: int | None = None) -> sp.csr_matrix:
    """Parse a whitespace ``row col value`` triplet matrix into a sparse array.

    CalculiX ``*MATRIX OUTPUT`` and several other solvers emit this shape.  DOF
    numbering is 1-based and only the lower triangle is stored, so the result is
    symmetrised here.
    """
    rows, cols, vals = [], [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError:
                continue
            rows.append(r - 1)
            cols.append(c - 1)
            vals.append(v)
    if n is None:
        n = max(max(rows), max(cols)) + 1
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    D = sp.diags(A.diagonal())
    return (A + A.T - D).tocsr()
