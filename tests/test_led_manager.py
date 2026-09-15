"""LedManager: connection state, max_brightness clamping, rebuild, and the
error-drops-to-ERROR-state contract (mirrors tests/test_plc_manager.py)."""

import pytest

from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.utilities.enums import ConnectionState
from core.utilities.exceptions import LedConnectionError


@pytest.fixture()
def stack() -> tuple[SimulatedLedClient, LedManager]:
    client = SimulatedLedClient()
    manager = LedManager(client, LedControllerSettings())
    manager.connect()
    return client, manager


def test_connect_transitions_to_connected(stack) -> None:
    client, manager = stack
    assert manager.state is ConnectionState.CONNECTED
    assert client.connected is True


def test_disconnect_transitions_to_disconnected(stack) -> None:
    client, manager = stack
    manager.disconnect()
    assert manager.state is ConnectionState.DISCONNECTED
    assert client.connected is False


def test_send_channel_builds_the_documented_command(stack) -> None:
    client, manager = stack
    response = manager.send_channel(1, 200)
    assert response == "!"
    assert client.sent == ["SA0200#"]


def test_send_channel_clamps_to_max_brightness() -> None:
    client = SimulatedLedClient()
    manager = LedManager(client, LedControllerSettings(max_brightness=200))
    manager.connect()
    manager.send_channel(1, 255)
    assert client.sent == ["SA0200#"]


def test_send_all_channels_sends_each_channel_in_order(stack) -> None:
    client, manager = stack
    responses = manager.send_all_channels(128)
    assert client.sent == ["SA0128#", "SB0128#", "SC0128#", "SD0128#"]
    assert responses == {1: "!", 2: "!", 3: "!", 4: "!"}


def test_send_multichannel_documented_example(stack) -> None:
    client, manager = stack
    manager.send_multichannel([(100, True), (128, True), (25, False), (0, True)])
    assert client.sent == ["S100T128T025F000TC#"]


def test_send_raw_is_not_clamped(stack) -> None:
    client, manager = stack
    manager.send_raw("SA0999#")  # deliberately malformed - raw tester must not "fix" it
    assert client.sent == ["SA0999#"]


def test_rebuild_swaps_client_and_settings_in_place(stack) -> None:
    old_client, manager = stack
    new_client = SimulatedLedClient()
    new_settings = LedControllerSettings(max_brightness=100)

    manager.rebuild(new_client, new_settings)

    assert manager.settings is new_settings
    assert manager.state is ConnectionState.DISCONNECTED
    assert old_client.connected is False
    new_client.connect()
    manager.send_channel(1, 255)
    assert new_client.sent == ["SA0100#"]  # new ceiling enforced


def test_comm_error_drops_to_error_state_and_disconnects(stack) -> None:
    client, manager = stack
    client.disconnect()  # simulate a lost link underneath the manager

    with pytest.raises(LedConnectionError):
        manager.send_channel(1, 100)

    assert manager.state is ConnectionState.ERROR
    assert manager.last_error


def test_reconnect_after_error_restores_connected_state(stack) -> None:
    client, manager = stack
    client.disconnect()
    with pytest.raises(LedConnectionError):
        manager.send_channel(1, 100)

    manager.connect()
    assert manager.state is ConnectionState.CONNECTED
    manager.send_channel(1, 100)  # no longer raises
