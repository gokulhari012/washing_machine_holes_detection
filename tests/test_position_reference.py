"""Position centre: which PLC datum a camera's X/Y is encoded against.

The PLC publishes two candidate datums per camera axis — the screwing (servo)
home (``servo_home_positions``) and the capture position the gantry stood at
when the camera took its picture (``capture_positions``). Each camera's
calibration picks one (Calibration page, Step 6); the screwing home is the
default and what every station did before the choice existed. The screw-driver
compensation offset (Step 5) is added to the millimetres before encoding, so
it must apply identically with either datum — the end-to-end tests at the
bottom pin that.
"""

import logging
from types import SimpleNamespace

import pytest

from core.calibration.calibration_manager import CalibrationManager
from core.calibration.calibration_model import CameraCalibration
from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.enums import PlcResultCode, PositionReference
from models.app_state import AppState
from services.inspection_service import InspectionService

from tests.test_register_map import make_config
from tests.test_single_camera_inspection import (
    FakeCameraManager,
    FakeConfig,
    FakeShifts,
    FakeVision,
)

SCREW = PositionReference.SCREW_HOME
CAPTURE = PositionReference.CAPTURE_POSITION
HOME = 6000
CAPTURE_AT = 9000


@pytest.fixture()
def stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["scaling"] = {"position_scale_x": 100, "position_scale_y": 100}
    config["registers"]["servo_home_positions"] = {
        "1": {"x": 220, "y": 222},
        "2": {"x": 224, "y": 226},
    }
    config["registers"]["capture_positions"] = {
        "1": {"x": 240, "y": 242},
        "2": {"x": 244, "y": 246},
    }
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap, vision_complete_delay_ms=0)
    manager.connect()
    for address in (220, 222, 224, 226):
        client.set_register(address, HOME)
    for address in (240, 242, 244, 246):
        client.set_register(address, CAPTURE_AT)
    return client, manager, rmap


def _written(client: SimulatedPlc, rmap: RegisterMap, camera: int) -> tuple[int, int]:
    """The combined 32-bit X/Y the PC wrote to *camera*'s position registers."""
    x_address, y_address = rmap.camera_positions[camera]
    return (
        RegisterMap.join_dword(*client.read_registers(x_address, 2)),
        RegisterMap.join_dword(*client.read_registers(y_address, 2)),
    )


# --------------------------------------------------------------- register map
def test_capture_positions_default_to_empty() -> None:
    assert RegisterMap.from_config(make_config()).capture_positions == {}


def test_capture_positions_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["capture_positions"] = {"1": {"x": 240, "y": 242}}
    assert RegisterMap.from_config(config).capture_positions == {1: (240, 242)}


def test_the_shipped_configs_carry_capture_positions_for_every_camera() -> None:
    """config/defaults/ must mirror the live file's keys (Restore Defaults)."""
    import json
    from pathlib import Path

    for path in (Path("config/plc.json"), Path("config/defaults/plc.json")):
        rmap = RegisterMap.from_config(json.loads(path.read_text(encoding="utf-8")))
        assert sorted(rmap.capture_positions) == [1, 2, 3, 4], path


# ----------------------------------------------------------------- PLC writes
def test_the_screwing_home_is_the_default_datum(stack) -> None:
    client, manager, rmap = stack
    manager.write_inspection_output({1: (2.0, -1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD)
    assert _written(client, rmap, 1) == (HOME + 200, HOME - 100)


def test_capture_position_is_the_datum_when_chosen(stack) -> None:
    client, manager, rmap = stack
    manager.write_inspection_output(
        {1: (2.0, -1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD,
        references={1: CAPTURE},
    )
    assert _written(client, rmap, 1) == (CAPTURE_AT + 200, CAPTURE_AT - 100)


def test_each_camera_is_encoded_against_its_own_choice(stack) -> None:
    client, manager, rmap = stack
    manager.write_inspection_output(
        {1: (2.0, -1.0), 2: (2.0, -1.0)},
        {1: PlcResultCode.GOOD, 2: PlcResultCode.GOOD},
        PlcResultCode.GOOD,
        references={1: CAPTURE, 2: SCREW},
    )
    assert _written(client, rmap, 1) == (CAPTURE_AT + 200, CAPTURE_AT - 100)
    assert _written(client, rmap, 2) == (HOME + 200, HOME - 100)


def test_a_single_camera_write_honours_the_choice_too(stack) -> None:
    client, manager, rmap = stack
    manager.write_camera_inspection_output(1, (2.0, -1.0), PlcResultCode.GOOD, reference=CAPTURE)
    assert _written(client, rmap, 1) == (CAPTURE_AT + 200, CAPTURE_AT - 100)


def test_the_capture_position_is_read_live_every_write(stack) -> None:
    """The gantry may stop somewhere else for the next part — never cached."""
    client, manager, rmap = stack
    manager.write_camera_inspection_output(1, (1.0, 1.0), PlcResultCode.GOOD, reference=CAPTURE)
    client.set_register(240, 7000)
    client.set_register(242, 7000)
    manager.write_camera_inspection_output(1, (1.0, 1.0), PlcResultCode.GOOD, reference=CAPTURE)
    assert _written(client, rmap, 1) == (7100, 7100)


def test_capture_without_registers_falls_back_to_zero(stack) -> None:
    """Same rule as a missing servo home: plain mm x scale."""
    client, manager, rmap = stack
    assert 3 not in rmap.capture_positions
    manager.write_camera_inspection_output(3, (2.0, 1.0), PlcResultCode.GOOD, reference=CAPTURE)
    assert _written(client, rmap, 3) == (200, 100)


def test_no_hole_is_still_the_sentinel_with_a_capture_datum(stack) -> None:
    client, manager, rmap = stack
    manager.write_camera_inspection_output(1, None, PlcResultCode.NG, reference=CAPTURE)
    assert _written(client, rmap, 1) == (RegisterMap.NO_HOLE_RAW, RegisterMap.NO_HOLE_RAW)


def test_the_log_names_the_capture_datum(stack, caplog) -> None:
    _client, manager, _rmap = stack
    with caplog.at_level(logging.INFO, logger="wmhd.plc"):
        manager.write_camera_inspection_output(
            1, (2.0, -1.0), PlcResultCode.GOOD, reference=CAPTURE
        )
    line = next(m for m in caplog.messages if m.startswith("Camera 1 position"))
    assert f"capture {CAPTURE_AT}" in line
    assert "home" not in line


def test_a_contiguous_capture_pair_is_one_read(stack) -> None:
    client, manager, _rmap = stack
    reads: list[tuple[int, int]] = []
    original = client.read_registers

    def recording(address: int, count: int) -> list[int]:
        reads.append((address, count))
        return original(address, count)

    client.read_registers = recording
    assert manager.read_capture_position(1) == (CAPTURE_AT, CAPTURE_AT)
    assert reads == [(240, 4)]


# ------------------------------------------------------------ calibration
class FakeRepository:
    def __init__(self) -> None:
        self.saved: list = []

    def get_all_active(self) -> dict:
        return {}

    def save(self, row) -> int:
        self.saved.append(row)
        return len(self.saved)


def test_an_uncalibrated_camera_uses_the_screwing_home() -> None:
    assert CalibrationManager(FakeRepository()).position_reference(1) is SCREW


def test_save_persists_and_takes_effect_in_the_same_call() -> None:
    repository = FakeRepository()
    manager = CalibrationManager(repository)
    manager.save(CameraCalibration(camera_index=2, position_reference=CAPTURE))
    assert manager.position_reference(2) is CAPTURE
    assert repository.saved[-1].position_reference == "capture_position"


def test_row_round_trip_keeps_the_choice() -> None:
    original = CameraCalibration(camera_index=4, position_reference=CAPTURE)
    assert CameraCalibration.from_row(original.to_row()).position_reference is CAPTURE


@pytest.mark.parametrize("stored", [None, "", "garbage"])
def test_a_legacy_or_unreadable_value_reads_as_the_screwing_home(stored) -> None:
    row = CameraCalibration(camera_index=1).to_row()
    row.position_reference = stored
    assert CameraCalibration.from_row(row).position_reference is SCREW


def test_a_machine_model_switch_keeps_the_live_choice() -> None:
    """A rig fact, like the screw offset — a profile never carries it."""
    manager = CalibrationManager(FakeRepository())
    manager._calibrations[1] = CameraCalibration(camera_index=1, position_reference=CAPTURE)
    snapshot = CameraCalibration(camera_index=1, pixels_per_mm_x=8.0).to_dict()
    assert "position_reference" not in snapshot

    manager.apply_live(CameraCalibration.from_dict(1, snapshot))
    assert manager.position_reference(1) is CAPTURE


# ------------------------------------------------------- end to end, Step 5
def _station(stack, calibrations: CalibrationManager) -> InspectionService:
    """A real PlcManager under the pipeline; the hole always measures (1, 2) mm."""
    _client, manager, _rmap = stack
    calibration = SimpleNamespace(
        has=lambda index: True,
        evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
        screw_offset=calibrations.screw_offset,
        position_reference=calibrations.position_reference,
    )
    return InspectionService(
        FakeCameraManager(), FakeVision(), calibration, manager,
        SimpleNamespace(save_inspection=lambda cycle: 1), AppState(), FakeConfig(),
        FakeShifts(),
    )


@pytest.mark.parametrize(("reference", "datum"), [(SCREW, HOME), (CAPTURE, CAPTURE_AT)])
@pytest.mark.parametrize("single_camera", [False, True])
def test_screw_compensation_applies_with_either_centre(
    stack, reference, datum, single_camera
) -> None:
    """Step 5's offset rides the millimetres, so it lands on whichever datum
    Step 6 picked — for a full cycle and a per-camera trigger alike."""
    client, _manager, rmap = stack
    calibrations = CalibrationManager(FakeRepository())
    calibrations._calibrations[1] = CameraCalibration(
        camera_index=1,
        screw_compensation_enabled=True,
        screw_offset_x_mm=0.5,
        screw_offset_y_mm=-0.25,
        position_reference=reference,
    )
    service = _station(stack, calibrations)

    if single_camera:
        cycle = service.run_camera_inspection(camera_index=1, machine_number=7)
    else:
        cycle = service.run_inspection(machine_number=7)

    # (1.0 + 0.5) mm and (2.0 - 0.25) mm at scale 100, from the chosen datum
    assert _written(client, rmap, 1) == (datum + 150, datum + 175)
    # ...while the recorded measurement stays the true hole
    assert (cycle.cameras[1].x_mm, cycle.cameras[1].y_mm) == (1.0, 2.0)


def test_without_compensation_only_the_datum_changes(stack) -> None:
    client, _manager, rmap = stack
    calibrations = CalibrationManager(FakeRepository())
    calibrations._calibrations[1] = CameraCalibration(camera_index=1, position_reference=CAPTURE)
    _station(stack, calibrations).run_inspection(machine_number=7)

    assert _written(client, rmap, 1) == (CAPTURE_AT + 100, CAPTURE_AT + 200)
    # camera 2 is uncalibrated here, so it keeps the screwing home
    assert _written(client, rmap, 2) == (HOME + 100, HOME + 200)
