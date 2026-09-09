"""Per-camera gantry gating: inspect only the cameras the PLC has in position.

Covers the whole path the feature adds — register map parsing, the live read,
the two PLC write paths that deliberately leave a skipped camera's registers
alone, and both cycle kinds (global trigger and per-camera trigger).

The fakes are shared with :mod:`tests.test_single_camera_inspection`, which
already stands in for the camera manager, vision engine, config and PLC.
"""

import pytest

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.enums import InspectionResult, PlcResultCode
from core.utilities.exceptions import PlcReadError
from models.app_state import AppState
from services.inspection_service import InspectionService

from tests.test_register_map import make_config
from tests.test_single_camera_inspection import (
    FakeCameraManager,
    FakeConfig,
    FakeShifts,
    FakeVision,
    RecordingPlc,
)
from types import SimpleNamespace


# --------------------------------------------------------------- register map
def test_gantry_status_defaults_to_empty() -> None:
    """An older plc.json with no gantry_status block keeps loading, and every
    camera is then always inspected."""
    assert RegisterMap.from_config(make_config()).gantry_status == {}


def test_gantry_status_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["gantry_status"] = {"1": 159, "2": 160}
    assert RegisterMap.from_config(config).gantry_status == {1: 159, 2: 160}


# ----------------------------------------------------------------- PLC reads
@pytest.fixture()
def stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128, "2": 129}
    config["registers"]["camera_triggers"] = {"1": 132, "2": 133}
    config["registers"]["camera_vision_complete"] = {"1": 136, "2": 137}
    config["registers"]["gantry_status"] = {"1": 159, "2": 160}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


def test_only_one_means_active(stack) -> None:
    """A garbled or half-written value must fail towards *not* inspecting."""
    client, manager, _rmap = stack

    client.set_register(159, RegisterMap.GANTRY_ACTIVE)
    assert manager.read_gantry_status(1) is True

    client.set_register(159, 0)
    assert manager.read_gantry_status(1) is False

    client.set_register(159, 7)
    assert manager.read_gantry_status(1) is False


def test_unconfigured_gantry_is_always_active(stack) -> None:
    """No register wired up -> the gate is inert, exactly as before it existed."""
    _client, manager, _rmap = stack
    assert manager.read_gantry_status(4) is True  # camera 4 has no register
    assert manager.gantry_status_configured(1) is True
    assert manager.gantry_status_configured(4) is False


def test_simulator_seeds_gantries_active(stack) -> None:
    """0 means "parked", so an unseeded simulator would inspect nothing at all
    on a station configured for the simulated protocol."""
    _client, manager, _rmap = stack
    assert manager.read_gantry_status(1) is True
    assert manager.read_gantry_status(2) is True


def test_the_shipped_configs_parse_and_stay_in_lockstep() -> None:
    """config/defaults/ must mirror every new key, or "Restore Defaults"
    silently drops the feature."""
    import json
    from pathlib import Path

    for path in (Path("config/plc.json"), Path("config/defaults/plc.json")):
        rmap = RegisterMap.from_config(json.loads(path.read_text(encoding="utf-8")))
        assert sorted(rmap.gantry_status) == [1, 2, 3, 4], path


# ---------------------------------------------------------------- PLC writes
def test_skipped_camera_registers_are_left_untouched(stack) -> None:
    """The last cycle that really inspected camera 2 still owns its registers —
    a deliberate skip must not overwrite them with a sentinel or an ERROR."""
    client, manager, rmap = stack
    x2_addr, y2_addr = rmap.camera_positions[2]
    client.set_register(x2_addr, 4321)  # left by an earlier real inspection
    client.set_register(y2_addr, 1234)
    client.set_register(129, int(PlcResultCode.GOOD))

    manager.write_inspection_output(
        positions={1: (1.0, 2.0)},
        camera_results={1: PlcResultCode.GOOD},
        result=PlcResultCode.GOOD,
        skipped={2},
    )

    assert client.read_registers(x2_addr, 1)[0] == 4321
    assert client.read_registers(y2_addr, 1)[0] == 1234
    assert client.read_registers(129, 1)[0] == int(PlcResultCode.GOOD)
    # ...while camera 1 and the overall handshake were written normally
    assert client.read_registers(128, 1)[0] == int(PlcResultCode.GOOD)
    assert client.read_registers(rmap.vision_complete, 1)[0] == 1


def test_absent_camera_still_gets_error_when_not_skipped(stack) -> None:
    """Not-inspected-because-of-a-fault keeps its old meaning: ERROR. Only an
    explicitly *skipped* camera is left alone."""
    client, manager, rmap = stack
    manager.write_inspection_output(
        positions={1: (1.0, 2.0)},
        camera_results={1: PlcResultCode.GOOD},
        result=PlcResultCode.ERROR,
    )
    assert client.read_registers(129, 1)[0] == int(PlcResultCode.ERROR)
    x2_addr, _y2 = rmap.camera_positions[2]
    assert client.read_registers(x2_addr, 1)[0] == RegisterMap.NO_HOLE_RAW


def test_skipped_camera_handshake_answers_without_claiming_a_result(stack) -> None:
    """A per-camera trigger raised for an inactive gantry still gets its
    handshake, so the PLC never dead-waits — but no result is invented."""
    client, manager, rmap = stack
    x_addr, y_addr = rmap.camera_positions[1]
    client.set_register(x_addr, 5000)
    client.set_register(y_addr, 6000)
    client.set_register(128, int(PlcResultCode.NG))
    client.set_register(132, 1)  # trigger the PLC raised

    manager.write_camera_skipped_output(1)

    assert client.read_registers(132, 1)[0] == 0  # trigger released
    assert client.read_registers(136, 1)[0] == 1  # completion raised
    assert client.read_registers(x_addr, 1)[0] == 5000  # untouched
    assert client.read_registers(y_addr, 1)[0] == 6000
    assert client.read_registers(128, 1)[0] == int(PlcResultCode.NG)


def test_skipped_handshake_releases_the_trigger_before_completion(stack) -> None:
    """Same ordering rule as a real per-camera cycle: the PLC may read the
    instant completion goes high, so the trigger must already be down."""
    client, manager, _rmap = stack
    client.set_register(132, 1)
    seen: list[tuple[int, int]] = []
    original = client.write_register

    def recording(address, value):
        original(address, value)
        if address in (132, 136):
            seen.append((address, value))

    client.write_register = recording  # type: ignore[method-assign]
    manager.write_camera_skipped_output(1)

    assert seen == [(132, 0), (136, 1)]


# --------------------------------------------------------------- the cycles
def _service(plc: RecordingPlc) -> tuple[InspectionService, FakeCameraManager, AppState]:
    cameras = FakeCameraManager()
    app_state = AppState()
    calibration = SimpleNamespace(
        has=lambda index: True,
        evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
        screw_offset=lambda index: (0.0, 0.0),
    )
    svc = InspectionService(
        cameras,
        FakeVision(),
        calibration,
        plc,
        SimpleNamespace(save_inspection=lambda cycle: 1),
        app_state,
        FakeConfig(),
        FakeShifts(),
    )
    return svc, cameras, app_state


def test_global_cycle_skips_the_inactive_camera() -> None:
    plc = RecordingPlc()
    plc.gantries = {2: False}
    svc, cameras, app_state = _service(plc)
    published: list[tuple[int, str]] = []
    app_state.camera_inspected.connect(
        lambda index, data: published.append((index, data.result.value))
    )

    cycle = svc.run_inspection(machine_number=7)

    assert cameras.capture_log == [1, 3, 4]  # camera 2 never grabbed
    assert cycle.cameras[2].result is InspectionResult.SKIPPED
    assert cycle.cameras[2].hole_found is False
    assert cycle.cameras[2].error == "gantry inactive"
    assert (2, "SKIPPED") in published  # the dashboard is told, not left stale

    positions, camera_results, overall, skipped = plc.full_writes[0]
    assert skipped == {2}
    assert 2 not in positions and 2 not in camera_results
    assert overall is PlcResultCode.GOOD  # a skip is not a failure


def test_a_skipped_camera_does_not_change_the_verdict() -> None:
    plc = RecordingPlc()
    plc.gantries = {1: False, 2: False, 3: False}
    svc, _cameras, _app_state = _service(plc)

    cycle = svc.run_inspection(machine_number=7)

    assert cycle.overall_result is InspectionResult.GOOD  # camera 4 alone judged
    assert {index: data.result.value for index, data in cycle.cameras.items()} == {
        1: "SKIPPED", 2: "SKIPPED", 3: "SKIPPED", 4: "GOOD",
    }


def test_a_cycle_with_every_gantry_inactive_is_an_error() -> None:
    """The PLC asked for a machine to be inspected and none of it was — that is
    a station problem, not a good part."""
    plc = RecordingPlc()
    plc.gantries = {index: False for index in (1, 2, 3, 4)}
    svc, cameras, _app_state = _service(plc)

    cycle = svc.run_inspection(machine_number=7)

    assert cameras.capture_log == []
    assert cycle.overall_result is InspectionResult.ERROR
    _positions, _results, overall, skipped = plc.full_writes[0]
    assert overall is PlcResultCode.ERROR
    assert skipped == {1, 2, 3, 4}


def test_a_failed_gantry_read_still_inspects_the_camera() -> None:
    """Losing a register read must never silently ship an un-inspected part."""
    plc = RecordingPlc()

    def boom(camera_index: int) -> bool:
        raise PlcReadError("link down")

    plc.read_gantry_status = boom
    svc, cameras, _app_state = _service(plc)

    cycle = svc.run_inspection(machine_number=7)

    assert cameras.capture_log == [1, 2, 3, 4]
    assert cycle.overall_result is InspectionResult.GOOD


# ------------------------------------------------- per-camera trigger cycle
def test_per_camera_trigger_on_an_inactive_gantry_skips_but_answers() -> None:
    plc = RecordingPlc()
    plc.gantries = {3: False}
    svc, cameras, app_state = _service(plc)
    app_state.set_counters(10, 8, 2)

    cycle = svc.run_camera_inspection(camera_index=3, machine_number=7)

    assert cameras.capture_log == []  # nothing captured
    assert plc.camera_writes == []  # no position, no result invented
    assert plc.skipped_writes == [3]  # handshake still answered
    assert cycle.overall_result is InspectionResult.SKIPPED
    assert cycle.cameras[3].result is InspectionResult.SKIPPED
    assert cycle.partial is True
    assert app_state.counters == (10, 8, 2)  # counters untouched


def test_per_camera_trigger_on_an_active_gantry_is_unaffected() -> None:
    plc = RecordingPlc()
    plc.gantries = {3: True}
    svc, cameras, _app_state = _service(plc)

    cycle = svc.run_camera_inspection(camera_index=3, machine_number=7)

    assert cameras.capture_log == [3]
    assert plc.skipped_writes == []
    assert plc.camera_writes == [(3, (1.0, 2.0), PlcResultCode.GOOD)]
    assert cycle.overall_result is InspectionResult.GOOD


def test_a_skipped_single_camera_cycle_is_still_recorded() -> None:
    """The skip has to be auditable — a part inspected by three of four cameras
    must be visibly that in the database, not silently short."""
    plc = RecordingPlc()
    plc.gantries = {1: False}
    saved: list = []
    cameras = FakeCameraManager()
    svc = InspectionService(
        cameras,
        FakeVision(),
        SimpleNamespace(
            has=lambda index: True,
            evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
        ),
        plc,
        SimpleNamespace(save_inspection=saved.append),
        AppState(),
        FakeConfig(),
        FakeShifts(),
    )

    cycle = svc.run_camera_inspection(camera_index=1, machine_number=7)

    assert saved == [cycle]
    assert cycle.shift == "Morning"
    assert cycle.serial_number == "WM-000007"
