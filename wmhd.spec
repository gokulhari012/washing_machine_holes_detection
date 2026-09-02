# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build recipe for the hole-detection station.

Layout produced by ``pyinstaller wmhd.spec`` (onedir — required, see below):

    dist/WMHoleDetection/
        WMHoleDetection.exe      launcher
        _internal/               Python runtime, Qt, OpenCV, resources/
        config/                  editable JSON config (copied by scripts/build_exe.py)
        data/ logs/ images/ backups/   created on first run, beside the .exe

Onedir rather than onefile: the app writes config/ and the SQLite database
next to the executable, an operator has to be able to edit config/*.json on
the station, and a onefile build re-extracts ~300 MB of Qt/OpenCV/pylon on
every launch. resources/ is read-only, so it is bundled inside _internal and
reached through ``sys._MEIPASS`` (see ui/theme.py).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

BASE_DIR = Path(SPECPATH)

# pypylon ships the Basler GenICam runtime as loose DLLs that no standard hook
# knows about; collect_all pulls the binaries and the .zip GenICam payload.
pylon_datas, pylon_binaries, pylon_hidden = collect_all("pypylon")

a = Analysis(
    ["main.py"],
    pathex=[str(BASE_DIR)],
    binaries=pylon_binaries,
    datas=[("resources", "resources")] + pylon_datas,
    # The camera/PLC adapters are imported lazily inside factories; name them
    # explicitly so a driver switch in camera.json cannot hit a missing module.
    hiddenimports=[
        "core.camera.basler_camera",
        "core.camera.hikrobot_camera",
        "core.plc.modbus_client",
        "core.plc.slmp_client",
        "core.plc.simulated_plc",
        *pylon_hidden,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Only QtCore/QtGui/QtWidgets are used; dropping the rest of Qt and the
    # scientific stack keeps the build near 300 MB instead of ~700 MB.
    excludes=[
        "tkinter", "matplotlib", "scipy", "pandas", "IPython", "notebook",
        "pytest", "_pytest", "PyQt5", "PyQt6", "PySide2",
        "PySide6.Qt3DAnimation", "PySide6.Qt3DCore", "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput", "PySide6.Qt3DLogic", "PySide6.Qt3DRender",
        "PySide6.QtBluetooth", "PySide6.QtCharts", "PySide6.QtDataVisualization",
        "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets", "PySide6.QtNfc", "PySide6.QtPdf",
        "PySide6.QtPdfWidgets", "PySide6.QtPositioning", "PySide6.QtQml",
        "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
        "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSensors",
        "PySide6.QtSerialPort", "PySide6.QtSpatialAudio", "PySide6.QtSql",
        "PySide6.QtStateMachine", "PySide6.QtTest", "PySide6.QtTextToSpeech",
        "PySide6.QtWebChannel", "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets", "PySide6.QtWebSockets",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WMHoleDetection",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=False hides the terminal on the station. Flip to True to see
    # tracebacks from a start-up crash that happens before logging is up.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WMHoleDetection",
)
