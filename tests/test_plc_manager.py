"""PlcManager handshake and fault recovery against the simulated PLC."""

import logging
import threading
import time

import pytest

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.plc.plc_manager import DEFAULT_VISION_COMPLETE_DELAY_MS, _delay_seconds
from core.utilities.enums import ConnectionState, PlcResultCode
from core.utilities.exceptions import PlcConnectionError

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
    # 2 apart, not 1: each axis is a 32-bit (2-register) value.
    config["registers"]["servo_home_positions"] = {
        "1": {"x": 144, "y": 146},
        "2": {"x": 148, "y": 150},
    }
    config["scaling"]["position_scale_x"] = 100
    config["scaling"]["position_scale_y"] = 100
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


def test_serial_number_register_is_read_when_configured() -> None:
    config = make_config()
    config["registers"]["serial_number"] = 104
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()

    client.set_register(104, 4711)
    assert manager.read_serial_number() == 4711


def test_serial_number_is_inert_when_not_configured(stack) -> None:
    """No address, no I/O — the caller falls back to the machine number."""
    _client, manager, rmap = stack
    assert rmap.serial_number is None
    assert manager.read_serial_number() is None


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
    # Every value here fits in 16 bits, so the high word (base+1) stays 0.
    x1, y1 = rmap.camera_positions[1]
    assert client.get_register(x1) == 125
    assert client.get_register(x1 + 1) == 0  # X high word
    assert client.get_register(y1) == 0
    x2, _y2 = rmap.camera_positions[2]
    assert client.get_register(x2) == 0  # no-hole sentinel
    assert client.get_register(rmap.result) == int(PlcResultCode.NG)
    assert client.get_register(rmap.vision_complete) == 1


def test_position_is_written_relative_to_servo_home(servo_stack) -> None:
    """home 6000 + 2.0 mm x scale 100 -> 6200; the negative axis lands below home."""
    client, manager, rmap = servo_stack
    client.set_register(144, 6000)  # camera 1 X home, low word (high word 0)
    client.set_register(146, 4000)  # camera 1 Y home, low word (high word 0)

    manager.write_inspection_output(
        {1: (2.0, -3.5)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    x1, y1 = rmap.camera_positions[1]
    assert client.get_register(x1) == 6200
    assert client.get_register(y1) == 3650


def test_servo_home_is_re_read_for_every_write(servo_stack) -> None:
    """The PLC may move the axis between cycles, so home is never cached."""
    client, manager, rmap = servo_stack
    client.set_register(144, 6000)
    client.set_register(146, 6000)
    manager.write_inspection_output(
        {1: (1.0, 1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    x1, _y1 = rmap.camera_positions[1]
    assert client.get_register(x1) == 6100

    client.set_register(144, 9000)
    manager.write_inspection_output(
        {1: (1.0, 1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    assert client.get_register(x1) == 9100


def test_no_hole_writes_the_sentinel_not_servo_home(servo_stack) -> None:
    """A camera that found nothing must not look like a hole sitting at home."""
    client, manager, rmap = servo_stack
    client.set_register(144, 6000)
    client.set_register(146, 6000)
    manager.write_inspection_output({1: None}, {1: PlcResultCode.NG}, PlcResultCode.NG)
    x1, y1 = rmap.camera_positions[1]
    assert client.get_register(x1) == RegisterMap.NO_HOLE_RAW
    assert client.get_register(y1) == RegisterMap.NO_HOLE_RAW


def test_read_servo_home_is_zero_when_unconfigured(stack) -> None:
    """No servo-home block: encoding falls back to plain scaled millimetres."""
    client, manager, rmap = stack
    assert manager.read_servo_home(1) == (0, 0)
    manager.write_inspection_output(
        {1: (12.5, 3.2)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    x1, y1 = rmap.camera_positions[1]
    assert client.get_register(x1) == 125
    assert client.get_register(y1) == 32


def test_position_beyond_16_bits_splits_across_low_and_high_words(servo_stack) -> None:
    """The whole point of the 32-bit conversion: a raw value beyond 65535
    must not wrap, and must decode back correctly by combining both words."""
    client, manager, rmap = servo_stack
    client.set_register(144, 0)  # camera 1 X home low word
    client.set_register(145, 1)  # camera 1 X home high word -> home = 65536
    assert manager.read_servo_home(1) == (65536, 0)

    manager.write_inspection_output(
        {1: (10.0, 0.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    # home 65536 + 10.0 * 100 -> raw 66536 = (high 1, low 1000)
    x1, _y1 = rmap.camera_positions[1]
    assert client.get_register(x1) == 1000       # low word
    assert client.get_register(x1 + 1) == 1       # high word
    assert rmap.decode_position(
        RegisterMap.join_dword(client.get_register(x1), client.get_register(x1 + 1)),
        65536,
        axis="x",
    ) == pytest.approx(10.0)


def test_write_raw_dword_writes_low_then_high_in_one_transaction(stack) -> None:
    """The PLC page's manual 32-bit write: base gets the low word, base+1 the
    high word, and both go out together so the PLC never sees half a target."""
    client, manager, _rmap = stack
    manager.write_raw_dword(300, 6_000_000)  # 0x005B8D80
    assert client.get_register(300) == 0x8D80
    assert client.get_register(301) == 0x005B
    assert RegisterMap.join_dword(
        client.get_register(300), client.get_register(301)
    ) == 6_000_000


def test_write_raw_dword_is_suspended_by_pause(stack) -> None:
    """It goes through the same write path as every other outgoing register,
    so Pause Communication covers it too."""
    client, manager, _rmap = stack
    manager.pause()
    manager.write_raw_dword(300, 6_000_000)
    assert client.get_register(300) == 0
    assert client.get_register(301) == 0


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


def _record_writes(client: SimulatedPlc) -> list[tuple[int, list[int]]]:
    """Capture every outgoing write as (base address, words), in order.

    Both entry points are wrapped: a 32-bit position pair goes out through
    ``write_registers`` as one transaction, everything else a word at a time.
    """
    order: list[tuple[int, list[int]]] = []
    write_one, write_many = client.write_register, client.write_registers

    def one(address: int, value: int) -> None:
        order.append((address, [value]))
        write_one(address, value)

    def many(address: int, values) -> None:
        order.append((address, list(values)))
        write_many(address, values)

    client.write_register = one
    client.write_registers = many
    return order


def test_positions_are_written_before_the_result_and_vision_complete(results_stack) -> None:
    """Order is the contract, not just the final values: the PLC may read
    everything the instant vision_complete goes high, so every position and
    every result register must already hold this cycle's values by then."""
    client, manager, rmap = results_stack
    order = _record_writes(client)

    manager.write_inspection_output(
        {1: (2.0, -1.0), 2: None},
        {1: PlcResultCode.GOOD, 2: PlcResultCode.NG},
        PlcResultCode.NG,
    )

    addresses = [address for address, _words in order]
    position_bases = {x for x, _y in rmap.camera_positions.values()} | {
        y for _x, y in rmap.camera_positions.values()
    }
    last_position = max(i for i, a in enumerate(addresses) if a in position_bases)
    last_camera_result = max(i for i, a in enumerate(addresses) if a in (128, 129))
    result_at = addresses.index(rmap.result)
    complete_at = addresses.index(rmap.vision_complete)

    assert last_position < last_camera_result < result_at < complete_at
    # vision_complete is strictly last, with nothing written after it.
    assert complete_at == len(addresses) - 1
    assert order[complete_at] == (rmap.vision_complete, [1])


def test_a_single_camera_cycle_writes_its_position_before_its_handshake() -> None:
    """Same contract on the per-camera path, where the trigger release sits
    between the result and that camera's own vision_complete."""
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128}
    config["registers"]["camera_triggers"] = {"1": 132}
    config["registers"]["camera_vision_complete"] = {"1": 136}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    client.set_register(132, 1)
    order = _record_writes(client)

    manager.write_camera_inspection_output(1, (2.0, -1.0), PlcResultCode.GOOD)

    addresses = [address for address, _words in order]
    x_address, _y = rmap.camera_positions[1]
    assert addresses.index(x_address) < addresses.index(128) < addresses.index(132)
    assert addresses.index(132) < addresses.index(136)
    assert addresses[-1] == 136  # that camera's vision_complete, strictly last


def _timed_writes(client: SimulatedPlc) -> list[tuple[int, float]]:
    """Capture (address, monotonic timestamp) for every outgoing write."""
    stamps: list[tuple[int, float]] = []
    write_one, write_many = client.write_register, client.write_registers

    def one(address: int, value: int) -> None:
        stamps.append((address, time.monotonic()))
        write_one(address, value)

    def many(address: int, values) -> None:
        stamps.append((address, time.monotonic()))
        write_many(address, values)

    client.write_register = one
    client.write_registers = many
    return stamps


def test_vision_complete_is_held_off_until_the_data_has_settled(results_stack) -> None:
    """The write order alone does not mean the PLC's scan has picked the data
    up; the configured delay is the margin, taken just before the flag."""
    client, manager, rmap = results_stack
    manager._vision_complete_delay_s = 0.08
    stamps = _timed_writes(client)

    manager.write_inspection_output(
        {1: (2.0, -1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )

    times = dict(stamps)
    assert times[rmap.vision_complete] - times[rmap.result] >= 0.08
    # ...and the pause is *only* there, not between every write.
    assert times[rmap.result] - times[128] < 0.08


def test_a_single_camera_cycle_settles_before_its_own_vision_complete() -> None:
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128}
    config["registers"]["camera_triggers"] = {"1": 132}
    config["registers"]["camera_vision_complete"] = {"1": 136}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap, vision_complete_delay_ms=80)
    manager.connect()
    stamps = _timed_writes(client)

    manager.write_camera_inspection_output(1, (2.0, -1.0), PlcResultCode.GOOD)

    times = dict(stamps)
    assert times[136] - times[132] >= 0.08  # after the trigger release


def test_a_skipped_camera_is_answered_without_the_settle_delay() -> None:
    """Nothing was measured, so there is no data in flight to settle — the
    handshake should not pay for a delay it does not need."""
    config = make_config()
    config["registers"]["camera_triggers"] = {"1": 132}
    config["registers"]["camera_vision_complete"] = {"1": 136}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap, vision_complete_delay_ms=500)
    manager.connect()

    started = time.monotonic()
    manager.write_camera_skipped_output(1)

    assert time.monotonic() - started < 0.5
    assert client.get_register(136) == 1


def test_zero_delay_disables_the_wait(results_stack) -> None:
    """0 is a real choice, distinct from "not configured"."""
    client, manager, _rmap = results_stack
    manager._vision_complete_delay_s = _delay_seconds(0)
    assert manager._vision_complete_delay_s == 0.0

    started = time.monotonic()
    manager.write_inspection_output(
        {1: (0.0, 0.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    assert time.monotonic() - started < 0.05


def test_an_unconfigured_delay_takes_the_default(stack) -> None:
    _client, manager, _rmap = stack
    assert manager._vision_complete_delay_s == DEFAULT_VISION_COMPLETE_DELAY_MS / 1000.0


def test_the_settle_delay_does_not_block_the_heartbeat(results_stack) -> None:
    """It runs on the inspection thread holding no lock, so the poll thread
    keeps the PLC's watchdog fed straight through it."""
    client, manager, rmap = results_stack
    manager._vision_complete_delay_s = 0.3

    beats: list[float] = []
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            manager.toggle_heartbeat()
            beats.append(time.monotonic())
            stop.wait(0.02)

    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    try:
        manager.write_inspection_output(
            {1: (1.0, 1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
        )
    finally:
        stop.set()
        pulse.join(timeout=2.0)

    # The 300 ms pause would have starved a 20 ms heartbeat down to ~1 beat.
    assert len(beats) > 5
    assert client.get_register(rmap.vision_complete) == 1


def test_a_position_write_is_logged_with_the_whole_derivation(servo_stack, caplog) -> None:
    """A position is the one value on this link that is computed rather than
    copied, so the log has to show every input: millimetres, the servo home it
    was measured from, the scale, the combined raw target and the two words
    that go on the wire."""
    client, manager, rmap = servo_stack
    client.set_register(144, 6000)  # camera 1 X home, low word
    client.set_register(146, 6000)  # camera 1 Y home, low word
    x_address, y_address = rmap.camera_positions[1]

    with caplog.at_level(logging.INFO, logger="wmhd.plc"):
        manager.write_inspection_output(
            {1: (2.0, -1.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
        )

    line = next(m for m in caplog.messages if m.startswith("Camera 1 position"))
    assert "+2.000 mm" in line and "-1.000 mm" in line   # the measurement
    assert "home 6000" in line                           # the datum
    assert "x 100" in line                               # the scale applied
    assert "= 6200" in line and "= 5900" in line         # the encoded targets
    assert f"[{x_address}]=6200" in line                 # the words written
    assert f"[{y_address}]=5900" in line


def test_the_logged_words_include_the_high_word_of_a_large_target(servo_stack, caplog) -> None:
    """Above 65535 the value only makes sense as a pair, so both words have to
    be in the line."""
    client, manager, rmap = servo_stack
    client.set_register(144, 0)
    client.set_register(145, 1)  # camera 1 X home = 65536
    x_address, _y = rmap.camera_positions[1]

    with caplog.at_level(logging.INFO, logger="wmhd.plc"):
        manager.write_inspection_output(
            {1: (10.0, 0.0)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
        )

    line = next(m for m in caplog.messages if m.startswith("Camera 1 position"))
    assert "= 66536" in line                       # the combined raw target
    assert f"[{x_address}]=1000" in line           # low word
    assert f"[{x_address + 1}]=1" in line          # high word


def test_a_no_hole_write_is_logged_as_the_sentinel_not_as_a_measurement(stack, caplog) -> None:
    """The no-hole case writes a sentinel, not a position — the log must not
    read as though the station measured 0 mm."""
    _client, manager, rmap = stack
    x_address, y_address = rmap.camera_positions[2]

    with caplog.at_level(logging.INFO, logger="wmhd.plc"):
        manager.write_inspection_output({2: None}, {}, PlcResultCode.NG)

    line = next(m for m in caplog.messages if m.startswith("Camera 2 position"))
    assert "no hole" in line
    assert f"sentinel {RegisterMap.NO_HOLE_RAW}" in line
    assert f"X[{x_address},{x_address + 1}]" in line
    assert f"Y[{y_address},{y_address + 1}]" in line
    assert "mm" not in line  # nothing that looks like a measurement


def test_every_written_camera_gets_its_own_position_line(results_stack, caplog) -> None:
    """One line per camera, so a four-camera cycle is legible in the log."""
    _client, manager, _rmap = results_stack

    with caplog.at_level(logging.INFO, logger="wmhd.plc"):
        manager.write_inspection_output(
            {1: (1.0, 1.0), 2: None},
            {1: PlcResultCode.GOOD, 2: PlcResultCode.NG},
            PlcResultCode.NG,
        )

    lines = [m for m in caplog.messages if "position ->" in m]
    assert len(lines) == len(_rmap_cameras(manager))


def _rmap_cameras(manager: PlcManager) -> list[int]:
    return sorted(manager.register_map.camera_positions)


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


def test_error_state_and_reconnect(stack) -> None:
    client, manager, rmap = stack
    client.disconnect()
    with pytest.raises(PlcConnectionError):
        manager.read_trigger()
    assert manager.state is ConnectionState.ERROR
    assert manager.ensure_connected() is True
    assert manager.state is ConnectionState.CONNECTED


# --------------------------------------------------------------------- pause
def test_pause_suspends_register_writes(stack) -> None:
    client, manager, rmap = stack
    manager.pause()

    manager.clear_trigger()  # register write, not the heartbeat
    assert client.get_register(rmap.trigger) == 0  # simulator seeds triggers at 0

    client.set_register(rmap.trigger, 1)
    manager.clear_trigger()
    assert client.get_register(rmap.trigger) == 1  # write skipped, value unchanged


def test_pause_does_not_suspend_the_heartbeat(stack) -> None:
    client, manager, rmap = stack
    manager.pause()

    manager.toggle_heartbeat()
    assert client.get_register(rmap.heartbeat) == 1
    manager.toggle_heartbeat()
    assert client.get_register(rmap.heartbeat) == 0


def test_pause_suspends_coil_writes(stack) -> None:
    client, manager, _rmap = stack
    manager.pause()

    manager.write_raw_coil(5, True)
    assert client.get_coil(5) is False


def test_resume_lets_writes_through_again(stack) -> None:
    client, manager, rmap = stack
    manager.pause()
    client.set_register(rmap.trigger, 1)
    manager.clear_trigger()
    assert client.get_register(rmap.trigger) == 1  # still suppressed

    manager.resume()
    manager.clear_trigger()
    assert client.get_register(rmap.trigger) == 0


def test_pause_and_resume_are_idempotent(stack) -> None:
    _client, manager, _rmap = stack
    manager.pause()
    manager.pause()
    assert manager.paused is True
    manager.resume()
    manager.resume()
    assert manager.paused is False


def test_pause_notifies_subscribers(stack) -> None:
    _client, manager, _rmap = stack
    seen: list[bool] = []
    manager.subscribe_paused(seen.append)

    manager.pause()
    manager.pause()  # idempotent: no duplicate notification
    manager.resume()

    assert seen == [True, False]


def test_write_position_scales_each_axis_by_its_own_key() -> None:
    """End to end: an X/Y pair written with different per-axis scales must
    reach the registers scaled independently, not both by the X scale."""
    config = make_config()
    config["scaling"] = {"position_scale_x": 100, "position_scale_y": 10}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()

    manager.write_inspection_output(
        {1: (12.5, 12.5)}, {1: PlcResultCode.GOOD}, PlcResultCode.GOOD
    )
    x1, y1 = rmap.camera_positions[1]
    assert client.get_register(x1) == 1250  # 12.5 * 100
    assert client.get_register(y1) == 125   # 12.5 * 10
