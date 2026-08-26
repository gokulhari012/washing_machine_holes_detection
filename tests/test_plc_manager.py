"""PlcManager handshake and fault recovery against the simulated PLC."""

import pytest

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.enums import ConnectionState, PlcResultCode
from core.utilities.exceptions import ConfigurationError, PlcConnectionError

from tests.test_register_map import make_config


@pytest.fixture()
def stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    rmap = RegisterMap.from_config(make_config())
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
def jog_stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["camera_jog"] = {
        "step": 10,
        "registers": {"1": {"x": 120, "y": 121, "home_x": 5, "home_y": 6}},
    }
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


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
    assert client.get_register(110) == 10125
    assert client.get_register(111) == 9968
    assert client.get_register(112) == 0  # no-hole sentinel
    assert client.get_register(rmap.result) == int(PlcResultCode.NG)
    assert client.get_register(rmap.vision_complete) == 1


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


@pytest.mark.parametrize(
    "direction, expected",
    [("up", (0, -10)), ("down", (0, 10)), ("left", (-10, 0)), ("right", (10, 0))],
)
def test_jog_camera_moves_by_step_in_each_direction(jog_stack, direction, expected) -> None:
    client, manager, rmap = jog_stack
    client.set_register(120, 100)
    client.set_register(121, 100)
    new_x, new_y = manager.jog_camera(1, direction)
    dx, dy = expected
    assert (new_x, new_y) == (100 + dx, 100 + dy)
    assert client.get_register(120) == new_x
    assert client.get_register(121) == new_y


def test_jog_camera_clamps_at_register_bounds(jog_stack) -> None:
    client, manager, rmap = jog_stack
    client.set_register(120, 5)
    new_x, _ = manager.jog_camera(1, "left")  # 5 - 10 would go negative
    assert new_x == 0

    client.set_register(120, 65530)
    new_x, _ = manager.jog_camera(1, "right")  # 65530 + 10 would overflow uint16
    assert new_x == 65535


def test_jog_camera_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.jog_camera(2, "up")


def test_jog_camera_rejects_bad_direction(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.jog_camera(1, "sideways")


def test_home_camera_writes_configured_values(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    client.set_register(120, 999)
    client.set_register(121, 999)
    assert manager.home_camera(1) == (5, 6)
    assert client.get_register(120) == 5
    assert client.get_register(121) == 6


def test_home_camera_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.home_camera(2)


def test_jog_configured(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    assert manager.jog_configured(1) is True
    assert manager.jog_configured(2) is False


def test_read_camera_jog_position_reads_current_registers(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    client.set_register(120, 250)
    client.set_register(121, 340)
    assert manager.read_camera_jog_position(1) == (250, 340)


def test_read_camera_jog_position_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.read_camera_jog_position(2)


def test_set_camera_jog_position_writes_given_values(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    assert manager.set_camera_jog_position(1, 250, 340) == (250, 340)
    assert client.get_register(120) == 250
    assert client.get_register(121) == 340


def test_set_camera_jog_position_clamps_at_register_bounds(jog_stack) -> None:
    client, manager, _rmap = jog_stack
    new_x, new_y = manager.set_camera_jog_position(1, -5, 70000)
    assert (new_x, new_y) == (0, 65535)
    assert client.get_register(120) == 0
    assert client.get_register(121) == 65535


def test_set_camera_jog_position_rejects_unconfigured_camera(jog_stack) -> None:
    _client, manager, _rmap = jog_stack
    with pytest.raises(ConfigurationError):
        manager.set_camera_jog_position(2, 0, 0)


def test_error_state_and_reconnect(stack) -> None:
    client, manager, rmap = stack
    client.disconnect()
    with pytest.raises(PlcConnectionError):
        manager.read_trigger()
    assert manager.state is ConnectionState.ERROR
    assert manager.ensure_connected() is True
    assert manager.state is ConnectionState.CONNECTED
