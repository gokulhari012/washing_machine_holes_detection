"""The PLC page's register table reads and writes whole 32-bit positionals.

``camera_positions`` and ``servo_home_positions`` are double-word registers:
the address configured for an axis is the *base*, holding the low 16 bits,
with the high word at base+1 (core.plc.register_map). The monitor table used
to read and write that base address alone, so its Live Value column
under-reported any servo target above 65535 and its per-row Set wrote half a
target. These tests pin both halves of the fix, plus the fact that the extra
word costs no extra PLC round trip (the pair is adjacent, so the existing
read clustering already covers it).
"""

from __future__ import annotations

import copy
import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from core.plc import RegisterMap
from core.utilities.enums import ConnectionState
from models.app_state import AppState
from ui import theme
from ui.plc.plc_page import PlcPage

# 6_000_000 raw units = 0x005B8D80: low word 0x8D80, high word 0x005B. Above
# the old uint16 ceiling, so a low-word-only read cannot produce it by
# accident.
BIG_RAW = 6_000_000
BIG_LOW, BIG_HIGH = RegisterMap.split_dword(BIG_RAW)

PLC_CONFIG = {
    "connection": {"protocol": "simulated", "ip": "127.0.0.1", "port": 502, "unit_id": 1},
    "scaling": {"position_scale_x": 100, "position_scale_y": 100},
    "polling": {},
    "registers": {
        "trigger": 100,
        "machine_number": 101,
        "heartbeat": 102,
        "result": 118,
        "vision_complete": 119,
        "camera_positions": {
            "1": {"x": 200, "y": 202},
            "2": {"x": 204, "y": 206},
            "3": {"x": 208, "y": 210},
            "4": {"x": 212, "y": 214},
        },
        "servo_home_positions": {
            "1": {"x": 220, "y": 222},
            "2": {"x": 224, "y": 226},
            "3": {"x": 228, "y": 230},
            "4": {"x": 232, "y": 234},
        },
    },
}


class FakeAuth:
    """Admin, so the per-row write gate is open."""

    is_admin = True

    def subscribe(self, callback) -> None:  # pragma: no cover - never fired here
        pass


class FakePlcService:
    """Records every read span and every write, over a flat register file."""

    def __init__(self) -> None:
        self.registers: dict[int, int] = {}
        self.read_spans: list[tuple[int, int]] = []
        self.dword_writes: list[tuple[int, int]] = []
        self.word_writes: list[tuple[int, int]] = []
        self.state = ConnectionState.CONNECTED
        self.paused = False

    def get_config(self) -> dict:
        return copy.deepcopy(PLC_CONFIG)

    def read_register(self, address: int, count: int = 1) -> list[int]:
        self.read_spans.append((address, count))
        return [self.registers.get(address + offset, 0) for offset in range(count)]

    def read_coil(self, address: int, count: int = 1) -> list[bool]:
        return [False] * count

    def write_register(self, address: int, value: int) -> None:
        self.word_writes.append((address, value))
        self.registers[address] = value

    def write_dword_register(self, address: int, value: int) -> None:
        self.dword_writes.append((address, value))
        low, high = RegisterMap.split_dword(value)
        self.registers[address] = low
        self.registers[address + 1] = high

    def write_coil(self, address: int, value: bool) -> None:  # pragma: no cover
        pass

    def set_paused(self, paused: bool) -> None:  # pragma: no cover
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def page(qapp):
    """Build the page, then dispose of it while Qt is still alive.

    Two things have to be undone, or the widgets survive to the interpreter's
    final garbage collection and crash it (an access violation inside
    ``gc_collect_harder`` at pytest teardown, long after every test passed):

    * the page's ``LabeledLed`` registers a **bound method** with
      ``ui.theme``'s module-global observer list and never unsubscribes — by
      design, since in the real application a page is built once and lives as
      long as the window. Here that list would pin every page this module
      builds. Truncating it back to its previous length drops exactly the
      callbacks this page added.
    * the widget itself, deleted and collected here rather than at exit, so
      the destruction happens while the QApplication is still usable.

    Keeping the AppState referenced until after the widget is gone matters
    too: the page holds queued connections to its signals.
    """
    service = FakePlcService()
    state = AppState()
    observers_before = len(theme._observers)
    widget = PlcPage(state, service, FakeAuth())

    yield widget, service

    widget._timer.stop()
    del theme._observers[observers_before:]
    widget.setParent(None)
    widget.deleteLater()
    qapp.processEvents()
    del widget, state
    gc.collect()


def _row_named(widget: PlcPage, name: str) -> int:
    for row in range(widget._table.rowCount()):
        if widget._table.item(row, 0).text() == name:
            return row
    raise AssertionError(f"no row named {name!r}")


def _live_value(widget: PlcPage, row: int) -> str:
    return widget._table.item(row, 3).text()


def test_live_value_combines_both_words_of_a_positional_register(page):
    """The column shows the servo target, not its low 16 bits."""
    widget, service = page
    service.registers[200] = BIG_LOW
    service.registers[201] = BIG_HIGH

    widget._refresh_viewer()

    row = _row_named(widget, "Camera 1 X")
    assert _live_value(widget, row) == str(BIG_RAW)
    # The low word alone would have read as this, which is the bug.
    assert _live_value(widget, row) != str(BIG_LOW)


def test_servo_home_rows_are_32_bit_too(page):
    """Both positional blocks are double words, not just camera_positions."""
    widget, service = page
    service.registers[220] = BIG_LOW
    service.registers[221] = BIG_HIGH

    widget._refresh_viewer()

    assert _live_value(widget, _row_named(widget, "Camera 1 Camera Home X")) == str(BIG_RAW)


def test_sixteen_bit_rows_are_left_alone(page):
    """Only the positional blocks widened; every other row still reads one word."""
    widget, service = page
    service.registers[118] = 2

    widget._refresh_viewer()

    assert _live_value(widget, _row_named(widget, "Result")) == "2"
    assert widget._table.item(_row_named(widget, "Camera 1 Result"), 1).text() == "Holding"
    assert widget._table.item(_row_named(widget, "Camera 1 X"), 1).text() == "Holding 32-bit"


def test_high_word_costs_no_extra_round_trip(page):
    """base+1 sits inside the span the clustering already reads, so reading
    the pair must not add a transaction."""
    widget, service = page
    widget._refresh_viewer()
    spans = list(service.read_spans)

    covered = {
        address + offset for address, count in spans for offset in range(count)
    }
    for base in (200, 202, 204, 220, 222):
        assert base in covered and base + 1 in covered

    service.read_spans.clear()
    widget._refresh_viewer()
    assert list(service.read_spans) == spans  # stable, not growing


def test_row_set_writes_the_whole_double_word(page):
    """Set on a positional row writes low+high in one transaction."""
    widget, service = page
    row = _row_named(widget, "Camera 1 X")
    widget._row_value_spins[row].setValue(BIG_RAW)

    widget._on_row_set(row)

    assert service.dword_writes == [(200, BIG_RAW)]
    assert service.word_writes == []
    assert service.registers[200] == BIG_LOW
    assert service.registers[201] == BIG_HIGH


def test_row_set_on_a_plain_register_stays_a_single_word_write(page):
    widget, service = page
    row = _row_named(widget, "Result")
    widget._row_value_spins[row].setValue(3)

    widget._on_row_set(row)

    assert service.word_writes == [(118, 3)]
    assert service.dword_writes == []


def test_positional_set_value_spin_accepts_more_than_16_bits(page):
    """A uint16-capped spin box could not express a real servo target."""
    widget, _service = page
    positional = widget._row_value_spins[_row_named(widget, "Camera 1 X")]
    plain = widget._row_value_spins[_row_named(widget, "Result")]

    assert positional.maximum() > 0xFFFF
    assert plain.maximum() == 0xFFFF


def test_pair_hint_follows_the_edited_base_address(page):
    """The tooltip naming the two registers must not go stale when the base
    address is re-pointed."""
    widget, _service = page
    row = _row_named(widget, "Camera 1 X")
    assert "200" in widget._table.item(row, 1).toolTip()

    widget._row_spins[row].setValue(6012)

    tooltip = widget._table.item(row, 1).toolTip()
    assert "6012" in tooltip and "6013" in tooltip
