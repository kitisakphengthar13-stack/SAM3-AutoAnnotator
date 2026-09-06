"""Workstation tokens and complete controls, including native subcontrols."""

from pathlib import Path
from PySide6.QtGui import QColor, QPalette

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
CHECK_ICON_PATH = (RESOURCE_DIR / "check.svg").as_posix()
PALETTE = {
    "app_bg": "#101419",
    "panel_bg": "#171d25",
    "panel_subtle": "#1e2630",
    "canvas_bg": "#0b0f14",
    "text": "#e4eaf2",
    "text_secondary": "#acb9c9",
    "text_muted": "#8796aa",
    "text_disabled": "#607084",
    "border_light": "#293340",
    "border": "#3b4b60",
    "primary": "#63d8bb",
    "primary_hover": "#84e5cd",
    "success": "#63d8bb",
    "danger": "#ff8c95",
    "selected_bg": "#203b3b",
    "selected_text": "#9cf0d9",
}


def apply_palette(widget):
    palette = widget.palette()
    for role, color in (
        (QPalette.Window, "#171d25"),
        (QPalette.WindowText, "#e4eaf2"),
        (QPalette.Base, "#11171e"),
        (QPalette.AlternateBase, "#1b232d"),
        (QPalette.Text, "#e4eaf2"),
        (QPalette.Button, "#202a36"),
        (QPalette.ButtonText, "#e4eaf2"),
        (QPalette.Highlight, "#284743"),
        (QPalette.HighlightedText, "#bbffeb"),
        (QPalette.ToolTipBase, "#253140"),
        (QPalette.ToolTipText, "#e4eaf2"),
        (QPalette.PlaceholderText, "#8796aa"),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#607084"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#607084"))
    widget.setPalette(palette)


APP_STYLESHEET = """
QWidget { color: #e4eaf2; font-family: "Segoe UI", "Inter", "DejaVu Sans"; font-size: 10pt; }
QMainWindow, QDialog { background: #101419; }
QMainWindow::separator { background: #101419; width: 5px; height: 5px; }
QMainWindow::separator:hover { background: #63d8bb; }
QWidget#datasetPanel, QWidget#annotationPanel, QWidget#setupPanel, QWidget#resultsPanel { background: #171d25; }
QWidget#canvasWorkspace, QWidget#toolRail, QWidget#canvasBar, QWidget#reviewBar { background: #171d25; }
QWidget#canvasBar { border-bottom: 1px solid #293340; }
QWidget#toolRail { border-right: 1px solid #293340; }
QWidget#reviewBar { border-top: 1px solid #293340; }
QToolBar#commandBar { background: #171d25; border-bottom: 1px solid #293340; padding: 8px 12px; spacing: 6px; }
QFrame#commandSeparator { background: #293340; }
QToolBar::separator { background: #293340; width: 1px; margin: 7px 6px; }
QMenuBar { background: #101419; color: #8796aa; padding: 2px 8px; }
QMenuBar::item { padding: 4px 9px; background: transparent; }
QMenuBar::item:selected { background: #293340; color: #e4eaf2; border-radius: 3px; }
QMenu { background: #1e2630; border: 1px solid #3b4b60; padding: 6px; }
QMenu::item { padding: 8px 28px 8px 12px; border-radius: 4px; }
QMenu::item:selected { background: #304139; color: #bbffeb; }
QMenu::item:disabled { color: #607084; }
QMenu::separator { height: 1px; background: #3b4b60; margin: 5px 8px; }
QMenu::indicator { width: 16px; height: 16px; }
QMenu::indicator:checked { image: url(@CHECK@); }
QTabWidget::pane { border: none; border-top: 1px solid #293340; background: #171d25; }
QTabBar::tab { background: #171d25; color: #8796aa; padding: 11px 24px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #63d8bb; border-bottom: 2px solid #63d8bb; }
QTabBar::tab:hover { color: #e4eaf2; }
QToolTip { background: #253140; color: #e4eaf2; border: 1px solid #4e6075; padding: 7px; }
QDockWidget { background: #171d25; titlebar-close-icon: url(@CLOSE@); titlebar-normal-icon: url(@FULL@); }
QDockWidget::title { background: #171d25; color: #acb9c9; padding: 9px 12px; font-weight: 600; }
QDockWidget::close-button, QDockWidget::float-button { border: none; padding: 3px; }
QDockWidget::close-button:hover, QDockWidget::float-button:hover { background: #304050; }
QLabel { background: transparent; border: none; }
QLabel#brand { font-size: 12pt; font-weight: 700; color: #63d8bb; }
QLabel#projectSubtitle, QLabel#mutedLabel, QLabel#canvasHint, QLabel#formLabel { color: #8796aa; }
QLabel#sectionTitle, QLabel#selectionTitle { font-weight: 600; font-size: 10pt; }
QLabel#eyebrow { color: #63d8bb; font-size: 9pt; font-weight: 700; }
QLabel#dialogTitle { font-size: 20pt; font-weight: 600; }
QLabel#validationError { color: #ff8c95; }
QLabel#selectionEmpty { color: #8796aa; padding: 14px 6px; }
QLabel#countBadge { color: #63d8bb; background: #203b3b; border-radius: 9px; padding: 3px 8px; }
QPushButton, QToolButton { background: #202a36; border: 1px solid #334050; border-radius: 5px; padding: 5px 10px; min-height: 24px; }
QPushButton:hover, QToolButton:hover { background: #2b3948; border-color: #52667d; }
QPushButton:pressed, QToolButton:pressed { background: #304b4a; }
QPushButton:focus, QToolButton:focus { border: 1px solid #63d8bb; }
QPushButton:checked, QToolButton:checked { color: #b0fbe5; background: #25433f; border-color: #63d8bb; }
QPushButton:disabled, QToolButton:disabled { color: #607084; background: #1a222c; border-color: #293340; }
QToolButton[iconOnly="true"] { padding: 5px; min-width: 24px; }
QToolButton#railButton { padding: 4px; min-height: 0; min-width: 0; border-color: transparent; background: transparent; }
QToolButton#railButton:hover { background: #283443; }
QToolButton#railButton:checked { background: #25433f; border: 1px solid #63d8bb; }
QWidget#dockTitle { background: #171d25; }
QToolButton#dockButton { background: transparent; border: 1px solid transparent; min-height: 0; min-width: 0; padding: 4px; }
QToolButton#dockButton:hover { background: #293b4c; border-color: #52667d; }
QToolButton#quietButton, QPushButton#quietButton { background: transparent; border-color: transparent; }
QToolButton#quietButton:hover, QPushButton#quietButton:hover { background: #293340; }
QPushButton#primaryButton, QToolButton#primaryButton, QToolButton#exportButton, QPushButton#emptyPrimary {
    background: #63d8bb; border-color: #63d8bb; color: #09251e; font-weight: 700;
}
QPushButton#primaryButton:hover, QToolButton#primaryButton:hover, QToolButton#exportButton:hover, QPushButton#emptyPrimary:hover { background: #84e5cd; }
QPushButton#primaryButton:disabled, QToolButton#primaryButton:disabled, QToolButton#exportButton:disabled { background: #243730; border-color: #2b443c; color: #75978c; }
QToolButton#dangerButton { color: #ff8c95; background: transparent; }
QToolButton#dangerButton:hover { background: #472c35; }
QToolButton#assistMenu { padding: 4px; min-height: 0; min-width: 0; }
QToolButton#assistMenu::menu-indicator { image: none; }
QToolButton::menu-indicator { image: url(@DOWN@); width: 12px; height: 12px; subcontrol-position: right center; right: 6px; }
QToolButton[popup="true"] { padding-right: 24px; }
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox {
    background: #11171e; border: 1px solid #334050; border-radius: 5px; padding: 6px 8px; min-height: 22px;
    selection-background-color: #284743; selection-color: #bbffeb;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover, QComboBox:hover, QDoubleSpinBox:hover { border-color: #52667d; }
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: #63d8bb; }
QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled { color: #607084; background: #19212a; border-color: #293340; }
QComboBox { padding-right: 30px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 28px; border-left: 1px solid #334050; }
QComboBox::down-arrow { image: url(@DOWN@); width: 14px; height: 14px; }
QComboBox QAbstractItemView { background: #1e2630; color: #e4eaf2; border: 1px solid #52667d; outline: none; selection-background-color: #284743; }
QDoubleSpinBox { padding-right: 30px; }
QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 28px; border-left: 1px solid #334050; border-bottom: 1px solid #334050; border-top-right-radius: 5px; background: #202a36; }
QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 28px; border-left: 1px solid #334050; border-bottom-right-radius: 5px; background: #202a36; }
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #354757; }
QDoubleSpinBox::up-arrow { image: url(@UP@); width: 13px; height: 13px; }
QDoubleSpinBox::down-arrow { image: url(@DOWN@); width: 13px; height: 13px; }
QCheckBox { spacing: 8px; color: #acb9c9; }
QCheckBox::indicator { width: 17px; height: 17px; background: #11171e; border: 1px solid #52667d; border-radius: 4px; }
QCheckBox::indicator:checked { background: #284743; border-color: #63d8bb; image: url(@CHECK@); }
QCheckBox::indicator:hover { border-color: #63d8bb; }
QCheckBox:disabled { color: #607084; }
QCheckBox::indicator:disabled { background: #1a222c; border-color: #334050; }
QGroupBox { border: 1px solid #293340; border-radius: 7px; margin-top: 12px; padding: 16px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #acb9c9; }
QLabel#pathDisplay { background: #11171e; color: #acb9c9; border: 1px solid #293340; border-radius: 4px; padding: 7px; min-height: 20px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: #171d25; }
QWidget#panelActionFooter { background: #171d25; border-top: 1px solid #293340; }
QListView, QTableView { background: #171d25; border: none; outline: none; alternate-background-color: #171d25; }
QListView::item:selected, QTableView::item:selected { background: #203b3b; }
QHeaderView::section { background: #171d25; color: #8796aa; border: none; padding: 5px; }
QTableView { gridline-color: #293340; }
QScrollBar:vertical { background: #11171e; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3b4b60; border-radius: 4px; min-height: 28px; margin: 2px; }
QScrollBar:horizontal { background: #11171e; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #3b4b60; border-radius: 4px; min-width: 28px; margin: 2px; }
QScrollBar::handle:hover { background: #60758b; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QProgressBar { background: #293340; border: none; border-radius: 3px; height: 5px; min-height: 5px; max-height: 5px; color: transparent; }
QProgressBar::chunk { background: #63d8bb; border-radius: 3px; }
QStatusBar { background: #101419; color: #8796aa; border-top: 1px solid #293340; font-size: 9pt; }
QStatusBar::item { border: none; }
QWidget#metricCard { background: #202a36; border: 1px solid #334050; border-radius: 6px; }
QLabel#metricValue { color: #e4eaf2; font-size: 20pt; font-weight: 600; }
QLabel#exportWarning { color: #f4c078; background: #342c22; border: 1px solid #534333; border-radius: 6px; padding: 12px; }
QWidget#taskProgress { background: #1b2d2a; border-top: 1px solid #315249; }
QWidget#emptyState, QWidget#imageLoadError { background: #0b0f14; }
QLabel#emptyTitle { color: #e4eaf2; font-size: 26pt; font-weight: 600; }
QLabel#emptySubtitle, QLabel#errorStateDetail { color: #8796aa; }
QLabel#errorStateTitle { color: #ff8c95; font-size: 16pt; font-weight: 600; }
QPushButton#emptyPrimary, QPushButton#emptySecondary { padding: 10px 18px; }
"""
for token, name in [
    ("CHECK", "check"),
    ("DOWN", "down"),
    ("UP", "up"),
    ("CLOSE", "close"),
    ("FULL", "fullscreen"),
]:
    APP_STYLESHEET = APP_STYLESHEET.replace(
        f"@{token}@", (RESOURCE_DIR / f"{name}.svg").as_posix()
    )
