"""SQLAlchemy 2.x ORM models — one class per table.

Conventions
-----------
- Timestamps are stored in **local plant time** (single-site system; operators
  filter "Today"/"Yesterday" in the time they live in).
- Enum-like columns store the ``StrEnum`` values from ``core.utilities.enums``
  as plain strings so the database stays readable with any SQLite browser.
- ``inspections`` ↔ ``inspection_details`` is a 1→4 parent/child with
  DB-level ``ON DELETE CASCADE`` (works with bulk retention deletes because
  the engine enables ``PRAGMA foreign_keys=ON``).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def now_local() -> datetime:
    """Default timestamp factory (local plant time)."""
    return datetime.now()


class Base(DeclarativeBase):
    """Declarative base shared by every model."""


# --------------------------------------------------------------------------- #
# Inspection data (hot path)
# --------------------------------------------------------------------------- #
class Inspection(Base):
    """One row per trigger cycle — the parent record of a full inspection."""

    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, index=True)
    machine_number: Mapped[int] = mapped_column(Integer, index=True)
    serial_number: Mapped[str] = mapped_column(String(64), default="")
    overall_result: Mapped[str] = mapped_column(String(8), index=True)  # InspectionResult
    plc_cycle_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    detection_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    operator: Mapped[str] = mapped_column(String(64), default="")
    shift: Mapped[str] = mapped_column(String(16), default="")

    details: Mapped[list["InspectionDetail"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="InspectionDetail.camera_index",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<Inspection id={self.id} machine={self.machine_number} "
            f"result={self.overall_result} at={self.created_at}>"
        )


class InspectionDetail(Base):
    """Per-camera result — exactly four children per inspection."""

    __tablename__ = "inspection_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    camera_index: Mapped[int] = mapped_column(Integer)  # 1..4
    camera_name: Mapped[str] = mapped_column(String(64), default="")
    hole_found: Mapped[bool] = mapped_column(Boolean, default=False)
    x_px: Mapped[float] = mapped_column(Float, default=0.0)
    y_px: Mapped[float] = mapped_column(Float, default=0.0)
    x_mm: Mapped[float] = mapped_column(Float, default=0.0)
    y_mm: Mapped[float] = mapped_column(Float, default=0.0)
    deviation_mm: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    result: Mapped[str] = mapped_column(String(8))  # InspectionResult
    image_path: Mapped[str | None] = mapped_column(String(260), default=None)

    inspection: Mapped[Inspection] = relationship(back_populates="details")


# --------------------------------------------------------------------------- #
# Configuration mirrors (audit trail of what the JSON files held)
# --------------------------------------------------------------------------- #
class CameraConfiguration(Base):
    """Persisted camera settings, one row per camera index."""

    __tablename__ = "camera_configurations"
    __table_args__ = (UniqueConstraint("camera_index", name="uq_camera_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(64))
    driver: Mapped[str] = mapped_column(String(32))  # CameraDriver
    connection_id: Mapped[str] = mapped_column(String(128), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    exposure_us: Mapped[int] = mapped_column(Integer, default=10000)
    gain_db: Mapped[float] = mapped_column(Float, default=0.0)
    gamma: Mapped[float] = mapped_column(Float, default=1.0)
    brightness: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=1280)
    height: Mapped[int] = mapped_column(Integer, default=1024)
    roi_x: Mapped[int] = mapped_column(Integer, default=0)
    roi_y: Mapped[int] = mapped_column(Integer, default=0)
    roi_width: Mapped[int] = mapped_column(Integer, default=0)  # 0 = full frame
    roi_height: Mapped[int] = mapped_column(Integer, default=0)
    trigger_mode: Mapped[str] = mapped_column(String(16), default="software")  # TriggerMode
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, onupdate=now_local)


class PlcConfiguration(Base):
    """Persisted PLC connection settings and register map (single row)."""

    __tablename__ = "plc_configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ip: Mapped[str] = mapped_column(String(45), default="192.168.0.10")
    port: Mapped[int] = mapped_column(Integer, default=502)
    protocol: Mapped[str] = mapped_column(String(32), default="modbus_tcp")
    unit_id: Mapped[int] = mapped_column(Integer, default=1)
    timeout_ms: Mapped[int] = mapped_column(Integer, default=1000)
    poll_interval_ms: Mapped[int] = mapped_column(Integer, default=50)
    trigger_register: Mapped[int] = mapped_column(Integer, default=100)
    machine_number_register: Mapped[int] = mapped_column(Integer, default=101)
    heartbeat_register: Mapped[int] = mapped_column(Integer, default=102)
    cam1_x_register: Mapped[int] = mapped_column(Integer, default=110)
    cam1_y_register: Mapped[int] = mapped_column(Integer, default=111)
    cam2_x_register: Mapped[int] = mapped_column(Integer, default=112)
    cam2_y_register: Mapped[int] = mapped_column(Integer, default=113)
    cam3_x_register: Mapped[int] = mapped_column(Integer, default=114)
    cam3_y_register: Mapped[int] = mapped_column(Integer, default=115)
    cam4_x_register: Mapped[int] = mapped_column(Integer, default=116)
    cam4_y_register: Mapped[int] = mapped_column(Integer, default=117)
    result_register: Mapped[int] = mapped_column(Integer, default=118)
    vision_complete_register: Mapped[int] = mapped_column(Integer, default=119)
    position_scale: Mapped[int] = mapped_column(Integer, default=10)
    position_offset: Mapped[int] = mapped_column(Integer, default=10000)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, onupdate=now_local)


class SystemConfiguration(Base):
    """Generic key/value store for application-level settings."""

    __tablename__ = "system_configurations"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    description: Mapped[str] = mapped_column(String(256), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, onupdate=now_local)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
class Calibration(Base):
    """Pixel→mm calibration per camera. History is kept; one active row per camera."""

    __tablename__ = "calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_index: Mapped[int] = mapped_column(Integer, index=True)
    pixels_per_mm_x: Mapped[float] = mapped_column(Float, default=1.0)
    pixels_per_mm_y: Mapped[float] = mapped_column(Float, default=1.0)
    homography_json: Mapped[str | None] = mapped_column(Text, default=None)  # 3x3 row-major
    ref_point_x_mm: Mapped[float] = mapped_column(Float, default=0.0)
    ref_point_y_mm: Mapped[float] = mapped_column(Float, default=0.0)
    rms_error: Mapped[float] = mapped_column(Float, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    calibrated_by: Mapped[str] = mapped_column(String(64), default="")
    calibrated_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)


# --------------------------------------------------------------------------- #
# Logs & users
# --------------------------------------------------------------------------- #
class LogEntry(Base):
    """Persisted application log record (fed by the logging fan-out handler)."""

    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, index=True)
    level: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(16), index=True)  # LogSource
    message: Mapped[str] = mapped_column(Text)


class User(Base):
    """Login account for password-protected areas."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))  # salted PBKDF2, set by AuthService
    role: Mapped[str] = mapped_column(String(16))  # UserRole
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, default=None)
