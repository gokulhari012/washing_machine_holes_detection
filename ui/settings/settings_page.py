"""Settings page: general options, shift rota, storage/backup, change password.

Reachable only after an administrator (or a developer, whose role covers
admin) logs in via the toolbar — the page is registered with
``min_role=UserRole.ADMIN`` in the nav rail, see ``MainWindow.add_page`` — so
unlike the PLC/Cameras pages it does not need its own "not logged in" gate
for the general and backup groups. When
``security.settings_password_protected`` is disabled, editing is allowed
without that session too. First run ships default ``admin``/``admin`` and
``developer``/``developer`` accounts — change them here.

The Shift Schedule group configures the three-shift rota. With "set the shift
automatically" ticked — the shipped setting — the shift stamped on every
inspection follows the clock, and the General group's manual Shift dropdown
becomes a read-only display of what the rota currently resolves to. Untick it
and the dropdown is the source of truth again, which is what the station did
before the rota existed. Either way the value reaches the rest of the
application through ``ShiftService`` and ``AppState``, never by a page reading
``application.shift`` for itself.

Editing the rota is validated *before* it is saved. Only two things are
refused outright: an unparsable time and a zero-length shift (start equal to
end, which could mean "never" or "all day" and so cannot be guessed at).

Everything else is legal but is annotated, because these are the ways a rota
can read correctly and still not do what was meant — each shift row shows its
own span ("8 h", or "22 h · wraps midnight" for one typed with its start after
its end), and ``_rota_notes`` reports, worst first:

* a shift **fully masked** by one listed above it, which can never be selected
  — the usual symptom of an inverted start/end;
* an **overlap**, naming the window and which shift wins it (the one listed
  first). This used to pass with no comment at all, which made it the most
  dangerous of the three;
* a **gap**, whose hours fall back to the manually selected shift.

The notes appear live under the group as you type and again in the Save
confirmation, which asks rather than blocks.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QTime
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
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from core.utilities import ConfigManager
from core.utilities.exceptions import ConfigurationError, VisionSystemError
from core.utilities.shift_schedule import Shift, ShiftSchedule, format_clock
from services.auth_service import AuthService
from services.backup_service import BackupService
from services.shift_service import ShiftService
from ui.theme import COLOR_WARN


def _to_qtime(value) -> QTime:
    return QTime(value.hour, value.minute)


class ShiftRow:
    """The three editors for one shift, kept together so the page can read
    them back as a config entry.

    Not a QWidget: the fields are laid out directly into the group's form so
    the three rows' columns line up, and this object is only the handle onto
    them. ``shift_id`` is preserved untouched through an edit — it is the
    stable key a rename must not disturb.
    """

    def __init__(self, shift: Shift) -> None:
        self.shift_id = shift.id
        self.name = QLineEdit(shift.name)
        self.name.setMaxLength(32)
        self.name.setToolTip(
            "Stamped on every inspection produced in this shift, and shown on "
            "the dashboard. Renaming affects new records only."
        )
        self.start = QTimeEdit(_to_qtime(shift.start))
        self.end = QTimeEdit(_to_qtime(shift.end))
        for editor in (self.start, self.end):
            editor.setDisplayFormat("HH:mm")
            editor.setFixedWidth(72)
        self.start.setToolTip("First minute of the shift (inclusive)")
        self.end.setToolTip(
            "First minute of the *next* shift (exclusive) — an end earlier "
            "than the start means the shift runs through midnight."
        )
        # Live read-out of what the two times actually add up to. This is the
        # only thing that makes an inverted start/end visible at a glance: a
        # shift typed as 08:00-06:00 is not an error, it is a legal 22 h
        # window, and "22 h · wraps midnight" says so where the times alone
        # look unremarkable.
        self.summary = QLabel()
        self.summary.setProperty("class", "dim")
        self.summary.setMinimumWidth(132)

    def widget(self) -> QWidget:
        """The three editors on one horizontal strip."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.name, stretch=1)
        layout.addWidget(self.start)
        layout.addWidget(QLabel("to"))
        layout.addWidget(self.end)
        layout.addWidget(self.summary)
        return row

    def describe(self, shift: Shift | None) -> None:
        """Show this row's span, or nothing while the row is mid-edit."""
        if shift is None:
            self.summary.setText("")
            return
        hours = shift.duration.total_seconds() / 3600.0
        text = f"{hours:g} h"
        if shift.wraps_midnight:
            text += "  ·  wraps midnight"
        self.summary.setText(text)

    def set_shift(self, shift: Shift) -> None:
        self.shift_id = shift.id
        self.name.setText(shift.name)
        self.start.setTime(_to_qtime(shift.start))
        self.end.setTime(_to_qtime(shift.end))

    def to_config(self) -> dict[str, str]:
        return {
            "id": self.shift_id,
            "name": self.name.text().strip(),
            "start": self.start.time().toString("HH:mm"),
            "end": self.end.time().toString("HH:mm"),
        }


class SettingsPage(QWidget):
    """General / storage / security configuration."""

    def __init__(
        self,
        config_manager: ConfigManager,
        auth_service: AuthService,
        backup_service: BackupService,
        shift_service: ShiftService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config_manager
        self._auth = auth_service
        self._backup = backup_service
        self._shifts = shift_service

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
        # Populated from the configured rota (not a fixed A/B/C), so the
        # manual choice can only ever be a shift that actually exists.
        self._shift = QComboBox()
        self._shift.setToolTip(
            "Which shift inspections are stamped with. Driven by the clock "
            "while the rota below is set to automatic."
        )
        self._serial_prefix = QLineEdit()
        general.addRow("Factory Name", self._factory)
        general.addRow("Operator Name", self._operator)
        general.addRow("Shift", self._shift)
        general.addRow("Serial Prefix", self._serial_prefix)
        left.addWidget(self._general_box)

        left.addWidget(self._build_shift_box())

        self._storage_box = QGroupBox("Storage && Backup")
        storage = QFormLayout(self._storage_box)
        self._save_images = QCheckBox("Save inspection images")
        self._save_images.setToolTip(
            "Write the annotated picture of every camera to images/<date>/."
        )
        self._ng_only = QCheckBox("Save NG images only")
        self._ng_only.setToolTip(
            "Skip pictures of cameras that judged GOOD; NG and ERROR are still saved."
        )
        # "NG only" filters the saving loop, so it means nothing while saving
        # is off entirely — grey it out rather than let it read as active.
        self._save_images.toggled.connect(self._ng_only.setEnabled)
        self._auto_backup = QCheckBox("Automatic daily backup")
        self._auto_backup.setToolTip(
            "Checked every 30 minutes; takes at most one backup per calendar day."
        )
        self._retention = QSpinBox()
        self._retention.setRange(0, 3650)
        self._retention.setSuffix(" days")
        self._retention.setToolTip("0 disables retention purging")
        backup_now = QPushButton("Backup Now")
        backup_now.clicked.connect(self._on_backup_now)
        storage.addRow("", self._save_images)
        storage.addRow("", self._ng_only)
        storage.addRow("", self._auto_backup)
        storage.addRow("", self._note(
            "Copies the whole inspection database (results, measurements and "
            "settings history) to backups/inspection_<date>_<time>.db. The "
            "newest 30 backup files are kept. Saved images are not included."
        ))
        storage.addRow("Retention", self._retention)
        storage.addRow("", self._note(
            "Permanently deletes inspection records and log entries older than "
            "this from the database. Saved image files are never deleted — "
            "clear images/ by hand. 0 keeps everything forever."
        ))
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

    # ----------------------------------------------------------- shift rota
    def _build_shift_box(self) -> QGroupBox:
        """The three-shift rota: automatic switch, three windows, live preview.

        The rows are built from whatever the config currently holds rather
        than from a hard-coded three, so a rota edited by hand to a different
        number of shifts still round-trips through this page instead of being
        silently truncated on the next Save.
        """
        self._shift_box = QGroupBox("Shift Schedule")
        layout = QVBoxLayout(self._shift_box)

        self._auto_shift = QCheckBox("Set the shift automatically from the time of day")
        self._auto_shift.setToolTip(
            "Unticked, the shift stays whatever is selected above until "
            "somebody changes it by hand."
        )
        self._auto_shift.toggled.connect(self._on_auto_shift_toggled)
        layout.addWidget(self._auto_shift)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        self._shift_rows: list[ShiftRow] = []
        for shift in self._shifts.schedule().shifts:
            row = ShiftRow(shift)
            row.name.textChanged.connect(self._on_shift_edited)
            row.start.timeChanged.connect(self._on_shift_edited)
            row.end.timeChanged.connect(self._on_shift_edited)
            self._shift_rows.append(row)
            form.addRow(f"Shift {len(self._shift_rows)}", row.widget())
        layout.addLayout(form)

        layout.addWidget(self._note(
            "Each shift runs from its start time up to (but not including) its "
            "end time, so consecutive shifts share a boundary without "
            "overlapping — an inspection at exactly 14:00 belongs to the shift "
            "starting then.\n"
            "• End earlier than start = the shift runs through midnight (the "
            "usual night shift, 22:00 to 06:00).\n"
            "• Two shifts covering the same hour is allowed — the one listed "
            "higher up wins it.\n"
            "• An hour no shift covers is allowed too — it falls back to the "
            "Shift selected in General, above."
        ))

        self._shift_preview = QLabel()
        self._shift_preview.setProperty("class", "dim")
        self._shift_preview.setWordWrap(True)
        layout.addWidget(self._shift_preview)

        # Gap / overlap / masked-shift notes. Separated from the preview and
        # given the warning colour because these are the three ways a rota can
        # be legal yet not do what the operator meant — an overlap in
        # particular used to pass completely unremarked.
        self._shift_warnings = QLabel()
        self._shift_warnings.setWordWrap(True)
        self._shift_warnings.setStyleSheet(f"color: {COLOR_WARN};")
        layout.addWidget(self._shift_warnings)
        return self._shift_box

    def _rota_notes(self, schedule: ShiftSchedule) -> list[str]:
        """The three ways a valid rota can still surprise you.

        Shared by the live label and the Save confirmation so the two can
        never disagree about what is wrong. Ordered worst first: a shift that
        can never be selected is a mistake, an overlap usually is, a gap is
        often deliberate.
        """
        notes: list[str] = []
        masked = schedule.unreachable()
        masked_ids = {shift.id for shift in masked}
        if masked:
            names = ", ".join(shift.name for shift in masked)
            notes.append(
                f"Never used: {names} — an earlier shift already covers that "
                f"whole window, so this shift can never be selected. Check for "
                f"a start time later than its end time."
            )
        for overlap in schedule.overlaps():
            # A shift that is already reported as never used overlaps the one
            # masking it at every minute it has, so listing those windows adds
            # nothing but noise on top of the note that actually diagnoses the
            # problem. One inverted start/end turned into four warnings that
            # buried it; only overlaps involving a still-usable shift survive.
            if all(shift.id in masked_ids for shift in overlap.shadowed):
                continue
            shadowed = ", ".join(shift.name for shift in overlap.shadowed)
            notes.append(
                f"Overlap {overlap.window_text}: {overlap.winner.name} and "
                f"{shadowed} both cover it — {overlap.winner.name} wins, "
                f"because it is listed first."
            )
        for start, end in schedule.coverage_gaps():
            notes.append(
                f"Uncovered {format_clock(start)}-{format_clock(end)}: "
                f"inspections then are stamped with the Shift selected in "
                f"General, above."
            )
        return notes

    def _form_schedule(self) -> ShiftSchedule:
        """The rota as currently typed, validated.

        Raises:
            ConfigurationError: a blank name, an unparsable time or a
                zero-length shift — surfaced to the operator on Save.
        """
        return ShiftSchedule.from_config(
            {
                "automatic": self._auto_shift.isChecked(),
                "schedule": [row.to_config() for row in self._shift_rows],
            }
        )

    def _on_shift_edited(self, *_args) -> None:
        """Re-sync the manual dropdown and the preview to what is typed.

        Runs on every keystroke, so it must never raise: a half-typed name is
        an invalid rota, and that is normal mid-edit — the preview simply says
        so and Save is where the operator is actually told.
        """
        try:
            schedule = self._form_schedule()
        except ConfigurationError as exc:
            self._shift_preview.setText(str(exc))
            self._shift_warnings.setText("")
            for row in self._shift_rows:
                row.describe(None)
            return
        self._sync_shift_choices(schedule)
        self._shift_preview.setText(self._preview_text(schedule))
        by_id = {shift.id: shift for shift in schedule.shifts}
        for row in self._shift_rows:
            row.describe(by_id.get(row.shift_id))
        notes = self._rota_notes(schedule)
        self._shift_warnings.setText("\n".join(f"⚠  {note}" for note in notes))
        self._shift_warnings.setVisible(bool(notes))

    def _sync_shift_choices(self, schedule: ShiftSchedule) -> None:
        """Refill the manual dropdown from the rota, keeping the selection.

        A shift the operator renames should stay selected, so the choice is
        preserved *by position* when its old name has disappeared — matching
        by name alone would silently jump the selection to another shift.
        """
        names = schedule.names
        if names == [self._shift.itemText(i) for i in range(self._shift.count())]:
            return
        position = self._shift.currentIndex()
        previous = self._shift.currentText()
        blocked = self._shift.blockSignals(True)
        self._shift.clear()
        self._shift.addItems(names)
        if previous in names:
            self._shift.setCurrentText(previous)
        elif 0 <= position < len(names):
            self._shift.setCurrentIndex(position)
        self._shift.blockSignals(blocked)

    def _preview_text(self, schedule: ShiftSchedule) -> str:
        """One line describing what the typed rota does, and any gap in it."""
        if not schedule.automatic:
            return "Automatic switching is off — the shift stays as selected above."
        now = datetime.now()
        current = schedule.shift_at(now)
        if current is None:
            head = "Right now the rota covers no shift"
        else:
            head = f"Right now: {current.name} ({current.window_text})"
        following = schedule.next_change_after(now)
        if following is not None:
            upcoming = schedule.shift_at(following)
            # Only announce a handover that actually hands over. In a rota
            # where one shift masks the next, the boundary still exists but the
            # same shift wins both sides of it, and "changes to Morning" while
            # already in Morning reads as a bug in the page rather than as the
            # symptom of a misconfigured rota that it is.
            if upcoming is not None and (current is None or upcoming.id != current.id):
                head += f", changes to {upcoming.name} at {format_clock(following.time())}"
        # Gaps, overlaps and masked shifts are reported by _rota_notes into the
        # warnings label below, rather than repeated here.
        return head

    def _on_auto_shift_toggled(self, checked: bool) -> None:
        """The manual dropdown is only an input while automatic is off; with
        it on the dropdown shows what the clock resolved to, read-only, so
        the two can never disagree on screen."""
        self._shift.setEnabled(not checked)
        if checked:
            self._apply_automatic_selection()
        self._on_shift_edited()

    def _apply_automatic_selection(self) -> None:
        """Point the dropdown at whatever the typed rota says it is now."""
        try:
            schedule = self._form_schedule()
        except ConfigurationError:
            return
        name = schedule.name_at(datetime.now())
        if name:
            blocked = self._shift.blockSignals(True)
            self._shift.setCurrentText(name)
            self._shift.blockSignals(blocked)

    @staticmethod
    def _note(text: str) -> QLabel:
        """Small wrapped caption explaining what a storage option actually does."""
        label = QLabel(text)
        label.setProperty("class", "dim")
        label.setWordWrap(True)
        return label

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
        self._serial_prefix.setText(application.get("serial_prefix", ""))

        # The rota must be in the form before the dropdown is filled from it.
        schedule = self._shifts.schedule()
        self._auto_shift.setChecked(schedule.automatic)
        by_id = {shift.id: shift for shift in schedule.shifts}
        for row in self._shift_rows:
            shift = by_id.get(row.shift_id)
            if shift is not None:
                row.set_shift(shift)
        self._sync_shift_choices(schedule)
        self._shift.setCurrentText(str(application.get("shift", "")))
        self._shift.setEnabled(not schedule.automatic)
        if schedule.automatic:
            self._apply_automatic_selection()
        self._shift_preview.setText(self._preview_text(schedule))
        self._save_images.setChecked(bool(storage.get("save_images", True)))
        self._ng_only.setChecked(bool(storage.get("save_ng_only", False)))
        self._ng_only.setEnabled(self._save_images.isChecked())
        self._auto_backup.setChecked(bool(database.get("auto_backup", True)))
        self._retention.setValue(int(database.get("retention_days", 90)))

    def _on_save(self) -> None:
        # Validate the rota before anything is written: a refused save must
        # leave app_config.json exactly as it was, not half-updated.
        try:
            schedule = self._form_schedule()
        except ConfigurationError as exc:
            QMessageBox.warning(self, "Shift Schedule", str(exc))
            return
        if schedule.automatic and not self._confirm_rota(schedule):
            return

        cfg = self._config.load("app_config")
        cfg.setdefault("application", {}).update(
            {
                "factory_name": self._factory.text().strip(),
                "operator_name": self._operator.text().strip(),
                "shift": self._shift.currentText(),
                "serial_prefix": self._serial_prefix.text(),
            }
        )
        cfg["shifts"] = schedule.to_config()
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
        # ShiftService drops its cache on this save through its own config
        # subscription and re-publishes; the page just re-renders its preview.
        self._on_shift_edited()
        QMessageBox.information(self, "Save Settings", "Settings saved.")

    def _confirm_rota(self, schedule: ShiftSchedule) -> bool:
        """Warn — but do not refuse — about a rota that is legal yet suspect.

        All three cases are configurations the code accepts and resolves
        deterministically, so none of them is an error: a plant that stops
        overnight has a legitimate gap, a handover overlap is a real practice,
        and a shift wrapping midnight is how the night shift is expressed. What
        they share is that the *typed times look fine* while the outcome may
        not be what was meant, so they are surfaced here rather than silently
        accepted. The same notes are already on the page as you type — this is
        the last chance to reconsider, not the first mention.
        """
        notes = self._rota_notes(schedule)
        if not notes:
            return True
        answer = QMessageBox.question(
            self,
            "Shift Schedule",
            "This rota will be saved and used, but check it first:\n\n"
            + "\n\n".join(f"•  {note}" for note in notes)
            + "\n\nSave anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # -------------------------------------------------------------- security
    def _protection_enabled(self) -> bool:
        return bool(
            self._config.get_value("app_config", "security.settings_password_protected", True)
        )

    def _apply_protection(self) -> None:
        allowed = self._auth.is_admin or not self._protection_enabled()
        for box in (self._general_box, self._shift_box, self._storage_box):
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
