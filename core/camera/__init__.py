"""Camera layer: abstract interface, driver adapters, manager, factory."""

from core.camera.camera_base import CameraBase, CameraSettings
from core.camera.camera_manager import CameraHealth, CameraManager
from core.camera.image_file_camera import ImageFileCamera
from core.camera.simulated_camera import SimulatedCamera
from core.camera.usb_camera import UsbCamera

from core.utilities.enums import CameraDriver
from core.utilities.exceptions import ConfigurationError

__all__ = [
    "CameraBase",
    "CameraSettings",
    "CameraHealth",
    "CameraManager",
    "ImageFileCamera",
    "SimulatedCamera",
    "UsbCamera",
    "create_camera",
]


def create_camera(settings: CameraSettings) -> CameraBase:
    """Factory: build the adapter named by ``settings.driver``.

    The HikRobot adapter is imported lazily so stations without the MVS SDK
    binding folder still start cleanly with simulated/USB cameras.
    """
    if settings.driver is CameraDriver.SIMULATED:
        return SimulatedCamera(settings)
    if settings.driver is CameraDriver.IMAGE_FILE:
        return ImageFileCamera(settings)
    if settings.driver is CameraDriver.USB:
        return UsbCamera(settings)
    if settings.driver is CameraDriver.HIKROBOT:
        from core.camera.hikrobot_camera import HikRobotCamera

        return HikRobotCamera(settings)
    # BASLER / DAHENG / IDS: add adapter modules and wire them here.
    raise ConfigurationError(
        f"No adapter implemented for camera driver {settings.driver.value!r}"
    )
