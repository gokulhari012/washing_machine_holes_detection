"""HikRobot (Hikvision industrial) camera adapter — integration skeleton.

The MVS SDK is machine-installed software, not a pip package, so this module
degrades gracefully: it imports cleanly everywhere and reports a clear error
if a HikRobot camera is selected without the SDK present.

Integration checklist
---------------------
1. Install HikRobot MVS (includes the GenICam runtime and drivers).
2. Copy the SDK's Python binding folder ``MvImport`` (from
   ``MVS/Development/Samples/Python``) next to this project or onto
   ``PYTHONPATH`` — it provides ``MvCameraControl_class``.
3. Set the camera's ``driver`` to ``"hikrobot"`` and ``connection_id`` to the
   device serial number in camera.json.

The method bodies below name the exact SDK calls to use; fill them in against
the SDK samples (``GrabImage.py`` is the closest reference).
"""

from __future__ import annotations

import numpy as np

from core.camera.camera_base import CameraBase, CameraSettings
from core.utilities.exceptions import CameraConnectionError

try:  # the SDK binding is optional at runtime
    from MvCameraControl_class import MvCamera  # type: ignore  # noqa: F401

    MVS_SDK_AVAILABLE = True
except ImportError:
    MVS_SDK_AVAILABLE = False

_SDK_HELP = (
    "HikRobot MVS SDK not available. Install MVS and put the SDK's "
    "'MvImport' Python bindings on PYTHONPATH (see hikrobot_camera.py header)."
)


class HikRobotCamera(CameraBase):
    """Adapter for HikRobot GigE/USB3 cameras via the MVS SDK."""

    def __init__(self, settings: CameraSettings) -> None:
        super().__init__(settings)
        self._handle = None  # MvCamera instance once implemented

    # ---------------------------------------------------------- driver hooks
    def _connect_device(self) -> None:
        if not MVS_SDK_AVAILABLE:
            raise CameraConnectionError(f"{self.name}: {_SDK_HELP}")
        # TODO(MVS): enumerate with MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE |
        #   MV_USB_DEVICE, device_list); match st_dev_info serial number against
        #   self._settings.connection_id; MV_CC_CreateHandle + MV_CC_OpenDevice;
        #   for GigE call MV_CC_GetOptimalPacketSize and set "GevSCPSPacketSize".
        raise CameraConnectionError(
            f"{self.name}: HikRobot adapter not implemented on this station yet"
        )

    def _disconnect_device(self) -> None:
        if self._handle is None:
            return
        # TODO(MVS): MV_CC_StopGrabbing, MV_CC_CloseDevice, MV_CC_DestroyHandle.
        self._handle = None

    def _apply_to_device(self, settings: CameraSettings) -> None:
        # TODO(MVS):
        #   MV_CC_SetFloatValue("ExposureTime", settings.exposure_us)
        #   MV_CC_SetFloatValue("Gain", settings.gain_db)
        #   MV_CC_SetFloatValue("Gamma", settings.gamma)
        #   MV_CC_SetIntValue("Width"/"Height", ...)
        #   MV_CC_SetEnumValue("TriggerMode", 1 if hardware trigger else 0)
        #   MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0)
        raise NotImplementedError

    def _grab(self) -> np.ndarray:
        # TODO(MVS): MV_CC_StartGrabbing once at connect; per frame use
        #   MV_CC_GetImageBuffer(st_frame, timeout_ms) -> convert pData to
        #   ndarray via np.ctypeslib, reshape to (height, width[, channels]),
        #   then MV_CC_FreeImageBuffer.
        raise NotImplementedError
