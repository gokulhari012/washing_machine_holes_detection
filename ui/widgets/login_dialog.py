"""Modal admin login prompt, opened from the main window's toolbar.

The single login surface for the whole app — engineering pages (Cameras,
PLC, Detection, Calibration, Settings) are hidden from the nav rail until an
administrator logs in here (see ``MainWindow``), so this dialog has to be
reachable regardless of role, unlike a login form embedded in one of the
pages it gates.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from core.database import User
from core.utilities.exceptions import VisionSystemError
from services.auth_service import AuthService


class LoginDialog(QDialog):
    """On accept, ``self.user`` holds the account that just logged in."""

    def __init__(self, auth_service: AuthService, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Administrator Login")
        self.setModal(True)
        self._auth = auth_service
        self.user: User | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._username = QLineEdit("admin")
        self._password = QLineEdit()
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._password.returnPressed.connect(self._on_accept)
        form.addRow("Username", self._username)
        form.addRow("Password", self._password)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        try:
            self.user = self._auth.login(self._username.text(), self._password.text())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Login", str(exc))
            return
        self.accept()
