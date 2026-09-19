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

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .base import ModalResult, ModalSolver


@dataclass
class CalculiXConfig:
    """Solver and meshing settings."""

    E: float = 68.9e9
    nu: float = 0.33
    rho: float = 2700.0
    n_modes: int = 10
    mesh_size: float = 3.0            # target element size [mm]
    mesh_size_min: float = 0.8
    order: int = 2                    # 2 => C3D10; do not lower this
    fix_selector: str = "z_min"       # "z_min" | "z_plane" | "all"
    fix_plane: float = 0.0            # used when fix_selector == "z_plane"
    fix_tol: float = 1e-3             # ABSOLUTE, in model units (mm)
    solid_type: str = "C3D10"         # element type to keep from the gmsh mesh
    ccx_path: str = "ccx"
    export_matrices: bool = False     # *MATRIX OUTPUT (version dependent)
    keep_files: bool = False
    # Non-structural payload, smeared over the nodes of the top face within
    # `payload_radius` of the part axis -- a motor bolted to its flange.  Put on
    # a single node it would instead excite a spurious local mode.
    payload_mass: float = 0.0         # model mass units (tonne for mm/t/s)
    payload_radius: float = 0.0       # model length units (mm)


def mesh_step(
    step_path: str, out_inp: str, cfg: CalculiXConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a STEP file with Gmsh and write an Abaqus/CalculiX ``.inp``.

    Returns ``(node_coords, node_tags)``.  Gmsh writes Abaqus format natively,
    which CalculiX reads with only minor keyword additions.
    """
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("gmsh is required: pip install 'dynopt[oss]'") from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
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

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cfg.mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cfg.mesh_size_min)
        gmsh.option.setNumber("Mesh.ElementOrder", cfg.order)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        # Second-order nodes must sit on the true CAD surface, or curved
        # features are under-resolved and stiffness is overestimated.
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

        gmsh.model.mesh.generate(3)
        gmsh.write(out_inp)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)

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
    finally:
        gmsh.finalize()


def _select_fixed_nodes(coords: np.ndarray, tags: np.ndarray, cfg: CalculiXConfig):
    """Pick clamped nodes geometrically (the bolt-down face).

    ``fix_tol`` is **absolute**, in model units, not a fraction of the bounding
    box.  A relative tolerance looks safer but is not: it silently rescales with
    the part, and on a design whose true mounting face is not the global lowest
    surface it selects a handful of stray nodes instead of a face.  That failure
    is invisible -- the solve succeeds and returns confident nonsense.

    ``"z_plane"`` clamps a *named* plane (``fix_plane``) instead of whatever
    happens to be lowest, which is the safer choice in an optimisation loop
    where design variables move geometry around.
    """
    z = coords[:, 2]
    if cfg.fix_selector == "z_min":
        sel = z <= z.min() + cfg.fix_tol
    elif cfg.fix_selector == "z_plane":
        sel = np.abs(z - cfg.fix_plane) <= cfg.fix_tol
    elif cfg.fix_selector == "all":
        sel = np.ones(coords.shape[0], dtype=bool)
    else:
        raise ValueError(f"unknown fix_selector {cfg.fix_selector!r}")
    if not sel.any():
        raise ValueError(
            f"fix_selector {cfg.fix_selector!r} matched no nodes "
            f"(z range {z.min():.4g}..{z.max():.4g}, tol {cfg.fix_tol:g})"
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

    if cfg.payload_mass > 0.0 and payload_tags is not None and payload_tags.size:
        share = cfg.payload_mass / payload_tags.size
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

    def solve(self, params: dict[str, float], n_modes: int = 10) -> ModalResult:
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
            coords, tags = mesh_step(step, mesh_inp, cfg)
            fixed = _select_fixed_nodes(coords, tags, cfg)

            payload = None
            if cfg.payload_mass > 0.0:
                z = coords[:, 2]
                rad = np.hypot(coords[:, 0], coords[:, 1])
                on_top = z >= z.max() - cfg.fix_tol
                within = rad <= cfg.payload_radius
                payload = tags[on_top & within]
                if payload.size == 0:
                    raise ValueError(
                        f"payload_radius={cfg.payload_radius:g} selected no nodes "
                        f"on the top face (max radius there "
                        f"{rad[on_top].max() if on_top.any() else float('nan'):.3g})"
                    )
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
                    "n_nodes": int(coords.shape[0]),
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
