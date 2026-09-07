"""PlcManager handshake and fault recovery against the simulated PLC."""

import pytest

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.enums import ConnectionState, PlcResultCode
from core.utilities.exceptions import CameraBusyError, ConfigurationError, PlcConnectionError

from tests.test_register_map import make_config


@pytest.fixture()
def stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    rmap = RegisterMap.from_config(make_config())
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


@pytest.fixture()
def servo_stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["registers"]["servo_home_positions"] = {
        "1": {"x": 144, "y": 145},
        "2": {"x": 146, "y": 147},
    }
    config["scaling"]["position_scale"] = 100
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


@pytest.fixture()
def results_stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128, "2": 129}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


@pytest.fixture()
def brightness_stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["registers"]["camera_brightness"] = {"1": 156, "2": 157}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


@pytest.fixture()
def jog_stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["camera_jog"] = {
        "step": 10,
        "registers": {"1": {"x": 120, "y": 121, "z": 122}},
    }
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


@pytest.fixture()
def jog_busy_stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["camera_jog"] = {
        "step": 10,
        "registers": {"1": {"x": 120, "y": 121, "z": 122, "busy": 200}},
    }
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


def test_rebuild_swaps_client_and_register_map_in_place(stack) -> None:
    old_client, manager, _rmap = stack
    new_config = make_config()
    new_config["registers"]["model_select"] = 103
    new_rmap = RegisterMap.from_config(new_config)
    new_client = SimulatedPlc(register_map=new_rmap)

    manager.rebuild(new_client, new_rmap)

    assert manager.register_map is new_rmap
    assert manager.state == ConnectionState.DISCONNECTED
    assert old_client.connected is False
    # the manager now talks to the new client/map, not the old one
    new_client.connect()
    new_client.set_register(103, 7)
    assert manager.read_model_select() == 7


def test_rebuild_clears_backoff_so_reconnect_is_immediate(stack) -> None:
    """A rebuild shouldn't leave the new client waiting out a backoff timer
    that was scheduled against the outgoing one."""
    _old_client, manager, rmap = stack
    new_client = SimulatedPlc(register_map=rmap)
    manager.rebuild(new_client, rmap)
    assert manager.ensure_connected() is True
    assert manager.state == ConnectionState.CONNECTED


def test_trigger_and_machine_number(stack) -> None:
    client, manager, rmap = stack
    assert manager.read_trigger() == 0
    client.fire_trigger(machine_number=4711)
    assert manager.read_trigger() == 1
    assert manager.read_machine_number() == 4711


def test_heartbeat_toggles(stack) -> None:
    client, manager, rmap = stack
    manager.toggle_heartbeat()
    assert client.get_register(rmap.heartbeat) == 1
    manager.toggle_heartbeat()
    assert client.get_register(rmap.heartbeat) == 0


def test_write_inspection_output(stack) -> None:
    client, manager, rmap = stack
    manager.write_inspection_output(
        {1: (12.5, -3.2), 2: None, 3: (0.0, 0.0), 4: (1.0, 2.0)},
        {1: PlcResultCode.GOOD, 2: PlcResultCode.NG, 3: PlcResultCode.GOOD, 4: PlcResultCode.GOOD},
        PlcResultCode.NG,
    )
    # No servo-home registers in this fixture, so home is 0 and the raw value
    # is just mm x scale. Negatives clamp at 0 without a home to sit below.
    assert client.get_register(110) == 125
    assert client.get_register(111) == 0
    assert client.get_register(112) == 0  # no-hole sentinel
    assert client.get_register(rmap.result) == int(PlcResultCode.NG)
    assert client.get_register(rmap.vision_complete) == 1


def test_position_is_written_relative_to_servo_home(servo_stack) -> None:
    """home 6000 + 2.0 mm x scale 100 -> 6200; the negative axis lands below home."""
    client, manager, _rmap = servo_stack
    client.set_register(144, 6000)  # camera 1 X home
    client.set_register(145, 4000)  # camera 1 Y home

    manager.write_inspection_output(
        {1: (2.0, -3.5)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    assert client.get_register(110) == 6200
    assert client.get_register(111) == 3650


def test_servo_home_is_re_read_for_every_write(servo_stack) -> None:
    """The PLC may move the axis between cycles, so home is never cached."""
    client, manager, _rmap = servo_stack
    client.set_register(144, 6000)
    client.set_register(145, 6000)
    manager.write_inspection_output(
        {1: (1.0, 1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    assert client.get_register(110) == 6100

    client.set_register(144, 9000)
    manager.write_inspection_output(
        {1: (1.0, 1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    assert client.get_register(110) == 9100


def test_no_hole_writes_the_sentinel_not_servo_home(servo_stack) -> None:
    """A camera that found nothing must not look like a hole sitting at home."""
    client, manager, _rmap = servo_stack
    client.set_register(144, 6000)
    client.set_register(145, 6000)
    manager.write_inspection_output({1: None}, {1: PlcResultCode.NG}, PlcResultCode.NG)
    assert client.get_register(110) == RegisterMap.NO_HOLE_RAW
    assert client.get_register(111) == RegisterMap.NO_HOLE_RAW


def test_read_servo_home_is_zero_when_unconfigured(stack) -> None:
    """No servo-home block: encoding falls back to plain scaled millimetres."""
    client, manager, _rmap = stack
    assert manager.read_servo_home(1) == (0, 0)
    manager.write_inspection_output(
        {1: (12.5, 3.2)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    assert client.get_register(110) == 125
    assert client.get_register(111) == 32


def test_write_inspection_output_writes_per_camera_results(results_stack) -> None:
    client, manager, _rmap = results_stack
    manager.write_inspection_output(
        {1: (12.5, -3.2), 2: None},
        {1: PlcResultCode.GOOD, 2: PlcResultCode.NG},
        PlcResultCode.NG,
    )
    assert client.get_register(128) == int(PlcResultCode.GOOD)
    assert client.get_register(129) == int(PlcResultCode.NG)


def test_write_inspection_output_defaults_missing_camera_result_to_error(results_stack) -> None:
    client, manager, _rmap = results_stack
    manager.write_inspection_output({1: (0.0, 0.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD)
    assert client.get_register(128) == int(PlcResultCode.GOOD)
    assert client.get_register(129) == int(PlcResultCode.ERROR)  # camera 2 not in camera_results


def test_clear_trigger_writes_zero(stack) -> None:
    client, manager, rmap = stack
    client.set_register(rmap.trigger, 1)
    manager.clear_trigger()
    assert client.get_register(rmap.trigger) == 0


def test_clear_camera_trigger_writes_zero_to_that_camera_only() -> None:
    config = make_config()
    config["registers"]["camera_triggers"] = {"1": 132, "2": 133}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()

    client.set_register(132, 1)
    client.set_register(133, 1)
    assert manager.clear_camera_trigger(1) is True
    assert client.get_register(132) == 0
    assert client.get_register(133) == 1  # untouched


def test_clear_camera_trigger_is_inert_when_unconfigured(stack) -> None:
    """No trigger register for that camera means no I/O, not an error."""
    _client, manager, _rmap = stack
    assert manager.clear_camera_trigger(1) is False


def test_write_camera_brightness_is_inert_when_unconfigured(stack) -> None:
    _client, manager, _rmap = stack
    assert manager.write_camera_brightness(1, 180) is False


def test_write_camera_brightness_writes_the_configured_register(brightness_stack) -> None:
    client, manager, _rmap = brightness_stack
    assert manager.write_camera_brightness(1, 180) is True
    assert client.get_register(156) == 180
    assert client.get_register(157) == 0  # camera 2 untouched


def test_write_camera_brightness_clamps_to_uint8_range(brightness_stack) -> None:
    client, manager, _rmap = brightness_stack
    manager.write_camera_brightness(1, -10)
    assert client.get_register(156) == 0
    manager.write_camera_brightness(1, 999)
    assert client.get_register(156) == 255


def test_camera_brightness_configured(brightness_stack, stack) -> None:
    assert brightness_stack[1].camera_brightness_configured(1) is True
    assert brightness_stack[1].camera_brightness_configured(9) is False
    assert stack[1].camera_brightness_configured(1) is False


def test_read_model_select_returns_none_when_unconfigured(stack) -> None:
    _client, manager, _rmap = stack
    assert manager.read_model_select() is None


def test_read_model_select_reads_the_configured_register() -> None:
    config = make_config()
    config["registers"]["model_select"] = 103
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()

    assert manager.read_model_select() == 0
    client.set_register(103, 7)
    assert manager.read_model_select() == 7


def test_write_model_select_is_inert_when_unconfigured(stack) -> None:
    _client, manager, _rmap = stack
    assert manager.write_model_select(5) is False


def test_write_model_select_writes_the_configured_register() -> None:
    config = make_config()
    config["registers"]["model_select"] = 103
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()

    assert manager.write_model_select(5) is True
    assert client.get_register(103) == 5


@pytest.mark.parametrize(
    "direction, register, expected",
    [
        ("x+", 120, 110),
        ("x-", 120, 90),
        ("y+", 121, 110),
        ("y-", 121, 90),
        ("z+", 122, 110),
        ("z-", 122, 90),
    ],
)
def test_jog_camera_moves_by_step_on_each_axis(jog_stack, direction, register, expected) -> None:
    client, manager, _rmap = jog_stack
    client.set_register(120, 100)
    client.set_register(121, 100)
    client.set_register(122, 100)
    new_value = manager.jog_camera(1, direction)
    assert new_value == expected
    assert client.get_register(register) == new_value


def test_jog_camera_clamps_at_register_bounds(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    client.set_register(120, 5)
    new_x = manager.jog_camera(1, "x-")  # 5 - 10 would go negative
    assert new_x == 0

    client.set_register(120, 65530)
    new_x = manager.jog_camera(1, "x+")  # 65530 + 10 would overflow uint16
    assert new_x == 65535


def test_jog_camera_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.jog_camera(2, "x+")


def test_jog_camera_rejects_bad_direction(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.jog_camera(1, "sideways")


def test_jog_camera_z_rejects_camera_without_z_register() -> None:
    config = make_config()
    config["camera_jog"] = {"step": 10, "registers": {"1": {"x": 120, "y": 121}}}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    with pytest.raises(ConfigurationError):
        manager.jog_camera(1, "z+")


def test_home_camera_zeroes_all_configured_axes(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    client.set_register(120, 999)
    client.set_register(121, 999)
    client.set_register(122, 999)
    assert manager.home_camera(1) == (0, 0, 0)
    assert client.get_register(120) == 0
    assert client.get_register(121) == 0
    assert client.get_register(122) == 0


def test_home_camera_skips_z_when_unconfigured() -> None:
    config = make_config()
    config["camera_jog"] = {"step": 10, "registers": {"1": {"x": 120, "y": 121}}}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    client.set_register(999, 42)  # untouched sentinel — no Z register to write
    assert manager.home_camera(1) == (0, 0, 0)
    assert client.get_register(120) == 0
    assert client.get_register(121) == 0
    assert client.get_register(999) == 42


def test_home_camera_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.home_camera(2)


def test_jog_configured(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    assert manager.jog_configured(1) is True
    assert manager.jog_configured(2) is False


def test_jog_z_configured(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    assert manager.jog_z_configured(1) is True
    assert manager.jog_z_configured(2) is False


def test_read_camera_jog_position_reads_current_registers(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    client.set_register(120, 250)
    client.set_register(121, 340)
    client.set_register(122, 430)
    assert manager.read_camera_jog_position(1) == (250, 340, 430)


def test_read_camera_jog_position_reports_zero_when_z_unconfigured() -> None:
    config = make_config()
    config["camera_jog"] = {"step": 10, "registers": {"1": {"x": 120, "y": 121}}}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    client.set_register(120, 250)
    client.set_register(121, 340)
    assert manager.read_camera_jog_position(1) == (250, 340, 0)


def test_read_camera_jog_position_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.read_camera_jog_position(2)


def test_set_camera_jog_position_writes_given_values(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    assert manager.set_camera_jog_position(1, 250, 340, 430) == (250, 340, 430)
    assert client.get_register(120) == 250
    assert client.get_register(121) == 340
    assert client.get_register(122) == 430


def test_set_camera_jog_position_drops_z_when_unconfigured() -> None:
    config = make_config()
    config["camera_jog"] = {"step": 10, "registers": {"1": {"x": 120, "y": 121}}}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    assert manager.set_camera_jog_position(1, 250, 340, 430) == (250, 340, 0)
    assert client.get_register(120) == 250
    assert client.get_register(121) == 340


def test_set_camera_jog_position_clamps_at_register_bounds(jog_stack) -> None:
    new_x, new_y, new_z = jog_stack[1].set_camera_jog_position(1, -5, 70000, 70000)
    assert (new_x, new_y, new_z) == (0, 65535, 65535)


def test_set_camera_jog_position_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.set_camera_jog_position(2, 0, 0)


# --------------------------------------------------- busy/moving coil gate
def test_jog_busy_configured(jog_stack, jog_busy_stack) -> None:
    assert jog_stack[1].jog_busy_configured(1) is False
    assert jog_busy_stack[1].jog_busy_configured(1) is True


def test_read_camera_jog_busy_false_when_unconfigured(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    assert manager.read_camera_jog_busy(1) is False


def test_jog_camera_raises_when_busy(jog_busy_stack) -> None:
    client, manager, _rmap = jog_busy_stack
    client.set_coil(200, True)
    with pytest.raises(CameraBusyError):
        manager.jog_camera(1, "x+")
    assert client.get_register(120) == 0  # refused before touching the position register


def test_jog_camera_raises_busy_coil_after_move(jog_busy_stack) -> None:
    client, manager, _rmap = jog_busy_stack
    assert client.get_coil(200) is False
    manager.jog_camera(1, "x+")
    assert client.get_coil(200) is True


def test_jog_camera_unaffected_by_busy_when_unconfigured(jog_stack) -> None:
    """jog_stack's camera 1 has no busy coil at all — the interlock must
    stay fully inert, exactly like before this feature existed."""
    client, manager, _rmap = jog_stack
    manager.jog_camera(1, "x+")  # would raise if busy-checking broke this path


def test_home_camera_raises_when_busy(jog_busy_stack) -> None:
    client, manager, _rmap = jog_busy_stack
    client.set_coil(200, True)
    with pytest.raises(CameraBusyError):
        manager.home_camera(1)


def test_home_camera_raises_busy_coil_after_move(jog_busy_stack) -> None:
    client, manager, _rmap = jog_busy_stack
    manager.home_camera(1)
    assert client.get_coil(200) is True


def test_set_camera_jog_position_raises_when_busy(jog_busy_stack) -> None:
    client, manager, _rmap = jog_busy_stack
    client.set_coil(200, True)
    with pytest.raises(CameraBusyError):
        manager.set_camera_jog_position(1, 10, 20, 30)
    assert client.get_register(120) == 0  # refused before touching any position register


def test_set_camera_jog_position_raises_busy_coil_after_move(jog_busy_stack) -> None:
    client, manager, _rmap = jog_busy_stack
    manager.set_camera_jog_position(1, 10, 20, 30)
    assert client.get_coil(200) is True


def test_busy_coil_cleared_by_plc_allows_next_move(jog_busy_stack) -> None:
    """Simulates the PLC finishing a move (clearing the coil back to 0) —
    the next command must be allowed again, not stay locked out forever."""
    client, manager, _rmap = jog_busy_stack
    manager.jog_camera(1, "x+")
    assert client.get_coil(200) is True
    client.set_coil(200, False)  # PLC-side reset
    manager.jog_camera(1, "x+")  # must not raise
    assert client.get_coil(200) is True


def test_error_state_and_reconnect(stack) -> None:
    client, manager, rmap = stack
    client.disconnect()
    with pytest.raises(PlcConnectionError):
        manager.read_trigger()
    assert manager.state is ConnectionState.ERROR
    assert manager.ensure_connected() is True
    assert manager.state is ConnectionState.CONNECTED
