"""The CLI, which had no tests at all.

It is the surface a user actually touches, and it is where the payload-node
defect lived: `coupling` fell back to node 0 when a backend reported no payload
nodes, which is every 3D result. On the motor mount node 0 is a clamped base
corner, so the command hung its dashpot on a constrained node and printed
coupling numbers for it -- no error, just a confident wrong answer.
"""

import json

import numpy as np
import pytest

from dynopt import cli
from dynopt.geometry import PART_NAMES


def run(argv, capsys):
    rc = cli.main(argv)
    return rc, capsys.readouterr().out


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
def test_unknown_part_is_a_clean_error_not_a_traceback():
    with pytest.raises(SystemExit):
        cli.main(["modal", "--part", "no_such_part"])


def test_unknown_backend_is_a_clean_error():
    with pytest.raises(SystemExit):
        cli.main(["modal", "--backend", "nonsense"])


@pytest.mark.parametrize("part", PART_NAMES)
def test_modal_runs_for_every_registered_part(part, capsys):
    """A part in the registry must work through the CLI, or it is not shipped."""
    rc, out = run(["modal", "--part", part, "--modes", "6"], capsys)
    assert rc == 0
    assert "mode  1" in out
    assert "nan" not in out.lower(), "a nan in CLI output means a missing mass"


def test_modal_accepts_a_params_override(capsys):
    _, base = run(["modal", "--modes", "4"], capsys)
    _, stiff = run(["modal", "--modes", "4", "--params",
                    json.dumps({"plate_thk": 0.008})], capsys)
    assert base != stiff, "--params had no effect"


def test_modal_reports_a_finite_mass(capsys):
    _, out = run(["modal", "--modes", "4"], capsys)
    line = [ln for ln in out.splitlines() if "mass:" in ln][0]
    grams = float(line.split("mass:")[1].split("g")[0])
    assert np.isfinite(grams) and grams > 0


# ---------------------------------------------------------------------------
# doe / coupling / optimize
# ---------------------------------------------------------------------------
def test_doe_writes_a_readable_archive(tmp_path, capsys):
    out = tmp_path / "doe.npz"
    rc, text = run(["doe", "-n", "4", "--modes", "4", "--out", str(out)], capsys)
    assert rc == 0 and out.exists()
    with np.load(out, allow_pickle=True) as z:
        assert z["X"].shape[0] == 4
        assert z["F"].shape[0] == 4
        assert len(z["names"]) == z["X"].shape[1]
    assert "availability" in text


def test_coupling_damps_the_payload_not_a_clamped_node(capsys):
    """The defect: absent payload nodes used to mean 'node 0', which is fixed."""
    from dynopt.fea.reference import ReferenceFrameSolver
    from dynopt.geometry import MOTOR_MOUNT

    res = ReferenceFrameSolver(MOTOR_MOUNT).solve(
        MOTOR_MOUNT.design_space.to_dict(MOTOR_MOUNT.design_space.defaults), n_modes=6
    )
    payload = res.meta["payload_nodes"]
    assert payload, "the reference backend must report its payload nodes"
    dof = cli._payload_dof(res, 0)
    assert dof // 6 == payload[0]
    # Whatever node that is, it must not be one of the clamped ones.
    frame = MOTOR_MOUNT.build_frame(
        MOTOR_MOUNT.design_space.to_dict(MOTOR_MOUNT.design_space.defaults)
    )
    assert payload[0] not in frame.fixed_nodes

    rc, out = run(["coupling", "--modes", "6"], capsys)
    assert rc == 0 and "coupled pairs" in out


def test_coupling_refuses_to_guess_when_the_payload_is_unknown():
    from dynopt.fea.base import ModalResult

    naked = ModalResult(frequencies=np.array([1.0]), mass=1.0,
                        meta={"backend": "forgetful"})
    with pytest.raises(SystemExit, match="no payload nodes"):
        cli._payload_dof(naked)


def test_optimize_runs_end_to_end_and_writes_its_result(tmp_path, capsys):
    out = tmp_path / "opt.json"
    rc, text = run([
        "optimize", "--modes", "6", "--init", "8", "--rounds", "1",
        "--per-round", "2", "--out", str(out),
    ], capsys)
    assert rc == 0 and out.exists()
    saved = json.loads(out.read_text())
    assert set(saved) >= {"params", "frequencies", "mass", "objective", "baseline"}
    assert np.isfinite(saved["objective"]), "a nan objective means a missing mass"
    assert saved["objective"] <= saved["baseline"]["objective"] + 1e-9, (
        "the optimiser returned something worse than the baseline"
    )
