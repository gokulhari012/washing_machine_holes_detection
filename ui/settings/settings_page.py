"""Settings page: general options, storage/backup, change password.

Reachable only after an administrator logs in via the toolbar (this page is
``admin_only`` in the nav rail — see ``MainWindow``), so unlike the PLC/
Cameras pages it does not need its own "not logged in" gate for the general
and backup groups. When ``security.settings_password_protected`` is
disabled, editing is allowed without that admin session too. First run ships
a default ``admin``/``admin`` account — change it here.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.utilities import ConfigManager
from core.utilities.exceptions import VisionSystemError
from services.auth_service import AuthService
from services.backup_service import BackupService


class SettingsPage(QWidget):
    """General / storage / security configuration."""

    def __init__(
        self,
        config_manager: ConfigManager,
        auth_service: AuthService,
        backup_service: BackupService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config_manager
        self._auth = auth_service
        self._backup = backup_service

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Settings")
        title.setProperty("class", "pageTitle")
        root.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        root.addLayout(columns)

        # ---------------------------------------------------------- general
        left = QVBoxLayout()
        self._general_box = QGroupBox("General")
        general = QFormLayout(self._general_box)
        self._factory = QLineEdit()
        self._operator = QLineEdit()
        self._shift = QComboBox()
        self._shift.addItems(["A", "B", "C"])
        self._serial_prefix = QLineEdit()
        general.addRow("Factory Name", self._factory)
        general.addRow("Operator Name", self._operator)
        general.addRow("Shift", self._shift)
        general.addRow("Serial Prefix", self._serial_prefix)
        left.addWidget(self._general_box)

        self._storage_box = QGroupBox("Storage && Backup")
        storage = QFormLayout(self._storage_box)
        self._save_images = QCheckBox("Save inspection images")
        self._ng_only = QCheckBox("Save NG images only")
        self._auto_backup = QCheckBox("Automatic daily backup")
        self._retention = QSpinBox()
        self._retention.setRange(0, 3650)
        self._retention.setSuffix(" days")
        self._retention.setToolTip("0 disables retention purging")
        backup_now = QPushButton("Backup Now")
        backup_now.clicked.connect(self._on_backup_now)
        storage.addRow("", self._save_images)
        storage.addRow("", self._ng_only)
        storage.addRow("", self._auto_backup)
        storage.addRow("Retention", self._retention)
        storage.addRow(backup_now)
        left.addWidget(self._storage_box)

        save_btn = QPushButton("Save Settings")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._on_save)
        self._save_btn = save_btn
        left.addWidget(save_btn)
        left.addStretch()
        columns.addLayout(left)

        # --------------------------------------------------------- security
        right = QVBoxLayout()
        self._session = QLabel()
        self._session.setProperty("class", "dim")
        right.addWidget(self._session)

        password_box = QGroupBox("Change Password")
        change = QFormLayout(password_box)
        self._old_password = QLineEdit()
        self._old_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._new_password = QLineEdit()
        self._new_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._confirm_password = QLineEdit()
        self._confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        change_btn = QPushButton("Change Password")
        change_btn.clicked.connect(self._on_change_password)
        change.addRow("Current", self._old_password)
        change.addRow("New", self._new_password)
        change.addRow("Confirm", self._confirm_password)
        change.addRow(change_btn)
        right.addWidget(password_box)
        right.addStretch()
        columns.addLayout(right)
        columns.addStretch()

        self._load()
        self._apply_protection()

    def showEvent(self, event) -> None:  # noqa: N802
        """Login now happens in the toolbar, outside this page — re-sync
        session state and the enabled/disabled boxes every time an admin
        navigates here, instead of only once at construction."""
        super().showEvent(event)
        self._apply_protection()

    # ------------------------------------------------------------ load/save
    def _load(self) -> None:
        cfg = self._config.load("app_config")
        application = cfg.get("application", {})
        database = cfg.get("database", {})
        storage = cfg.get("storage", {})
        self._factory.setText(application.get("factory_name", ""))
        self._operator.setText(application.get("operator_name", ""))
        self._shift.setCurrentText(application.get("shift", "A"))
        self._serial_prefix.setText(application.get("serial_prefix", ""))
        self._save_images.setChecked(bool(storage.get("save_images", True)))
        self._ng_only.setChecked(bool(storage.get("save_ng_only", False)))
        self._auto_backup.setChecked(bool(database.get("auto_backup", True)))
        self._retention.setValue(int(database.get("retention_days", 90)))

    def _on_save(self) -> None:
        cfg = self._config.load("app_config")
        cfg.setdefault("application", {}).update(
            {
                "factory_name": self._factory.text().strip(),
                "operator_name": self._operator.text().strip(),
                "shift": self._shift.currentText(),
                "serial_prefix": self._serial_prefix.text(),
            }
        )
        cfg.setdefault("storage", {}).update(
            {
                "save_images": self._save_images.isChecked(),
                "save_ng_only": self._ng_only.isChecked(),
            }
        )
        cfg.setdefault("database", {}).update(
            {
                "auto_backup": self._auto_backup.isChecked(),
                "retention_days": self._retention.value(),
            }
        )
        try:
            self._config.save("app_config", cfg)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save Settings", str(exc))
            return
        QMessageBox.information(self, "Save Settings", "Settings saved.")

    # -------------------------------------------------------------- security
    def _protection_enabled(self) -> bool:
        return bool(
            self._config.get_value("app_config", "security.settings_password_protected", True)
        )

    def _apply_protection(self) -> None:
        allowed = self._auth.is_admin or not self._protection_enabled()
        for box in (self._general_box, self._storage_box):
            box.setEnabled(allowed)
        self._save_btn.setEnabled(allowed)
        user = self._auth.current_user
        if user is not None:
            self._session.setText(f"Logged in as {user.username} ({user.role})")
        elif allowed:
            self._session.setText("Settings unprotected (security.settings_password_protected = false)")
        else:
            self._session.setText("Login required to edit settings — use Login in the toolbar")

    def _on_change_password(self) -> None:
        if self._new_password.text() != self._confirm_password.text():
            QMessageBox.warning(self, "Change Password", "New passwords do not match.")
            return
        if self._auth.current_user is None:
            QMessageBox.warning(
                self, "Change Password", "Log in first (toolbar Login button)."
            )
            return
        username = self._auth.current_user.username
        try:
            self._auth.change_password(
                username, self._old_password.text(), self._new_password.text()
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Change Password", str(exc))
            return
        for field in (self._old_password, self._new_password, self._confirm_password):
            field.clear()
        QMessageBox.information(self, "Change Password", "Password changed.")

    # --------------------------------------------------------------- backup
    def _on_backup_now(self) -> None:
        try:
            path = self._backup.backup_now()
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Backup", str(exc))
            return
        QMessageBox.information(self, "Backup", f"Backup created:\n{path}")
