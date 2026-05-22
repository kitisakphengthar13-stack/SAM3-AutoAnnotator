from pathlib import Path


CHECK_ICON_PATH = (Path(__file__).resolve().parent / "assets" / "check.svg").as_posix()


PALETTE = {
    "app_bg": "#f5f6f8",
    "panel_bg": "#ffffff",
    "panel_subtle": "#f8fafc",
    "canvas_bg": "#0f172a",
    "text": "#111827",
    "text_secondary": "#475569",
    "text_muted": "#64748b",
    "text_disabled": "#9ca3af",
    "border_light": "#e5e7eb",
    "border": "#cbd5e1",
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "success": "#15803d",
    "danger": "#dc2626",
    "selected_bg": "#dbeafe",
    "selected_text": "#1e3a8a",
}


APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {PALETTE["app_bg"]};
    color: {PALETTE["text"]};
    font-size: 9pt;
}}

QToolBar#commandBar {{
    background: {PALETTE["panel_bg"]};
    border-bottom: 1px solid {PALETTE["border_light"]};
    spacing: 3px;
    padding: 2px 6px;
}}
QToolBar#commandBar::separator {{
    background: {PALETTE["border_light"]};
    width: 1px;
    margin: 5px 5px;
}}
QLabel#appTitle {{
    font-size: 9.5pt;
    font-weight: 700;
    color: {PALETTE["text"]};
    background: transparent;
    border: none;
}}
QToolBar#commandBar QLabel {{
    background: transparent;
    border: none;
}}
QLabel#brandSeparator {{
    background: {PALETTE["border_light"]};
    margin: 4px 3px;
}}
QLabel#projectSubtitle {{
    color: {PALETTE["text_secondary"]};
    background: transparent;
    border: none;
}}
QLabel#mutedLabel {{
    color: {PALETTE["text_muted"]};
}}
QLabel#formLabel {{
    background: transparent;
    border: none;
    color: {PALETTE["text_secondary"]};
    padding: 0 4px 0 0;
    font-weight: 500;
}}
QLabel#sectionTitle {{
    font-size: 9.5pt;
    font-weight: 700;
    color: {PALETTE["text"]};
}}
QLabel#canvasHint {{
    background: {PALETTE["canvas_bg"]};
    color: #d1d5db;
    padding: 3px 8px;
    border-bottom: 1px solid #1e293b;
}}

QGroupBox {{
    background: {PALETTE["panel_bg"]};
    border: 1px solid {PALETTE["border_light"]};
    border-radius: 2px;
    margin-top: 9px;
    padding: 9px 7px 7px 7px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 7px;
    padding: 0 4px;
    color: {PALETTE["text_secondary"]};
}}
QGroupBox QLabel {{
    background: transparent;
    border: none;
}}

QPushButton, QToolButton {{
    background: {PALETTE["panel_bg"]};
    border: 1px solid {PALETTE["border_light"]};
    border-radius: 3px;
    padding: 3px 7px;
    min-height: 17px;
    color: {PALETTE["text"]};
}}
QPushButton:hover, QToolButton:hover {{
    background: {PALETTE["panel_subtle"]};
    border-color: {PALETTE["border"]};
}}
QPushButton:pressed, QToolButton:pressed {{
    background: #eef2f7;
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {PALETTE["text_disabled"]};
    background: #f1f3f5;
    border-color: {PALETTE["border_light"]};
}}
QPushButton#primaryButton {{
    background: #f8fbff;
    border-color: {PALETTE["border"]};
    color: {PALETTE["primary"]};
    font-weight: 700;
}}
QPushButton#primaryButton:hover {{
    background: #eff6ff;
    border-color: {PALETTE["primary"]};
    color: {PALETTE["primary_hover"]};
}}
QPushButton#primaryButton:disabled {{
    color: {PALETTE["text_disabled"]};
    background: #f1f3f5;
    border-color: {PALETTE["border_light"]};
}}
QPushButton#exportButton {{
    background: #f8fcf9;
    border-color: {PALETTE["border"]};
    color: {PALETTE["success"]};
    font-weight: 700;
}}
QPushButton#exportButton:hover {{
    background: #f0fdf4;
    border-color: {PALETTE["success"]};
}}
QPushButton#dangerButton {{
    background: {PALETTE["panel_bg"]};
    border-color: {PALETTE["border_light"]};
    color: {PALETTE["danger"]};
}}
QPushButton#dangerButton:hover {{
    background: #fff5f5;
    border-color: #fecaca;
}}
QToolButton#drawButton:checked {{
    background: {PALETTE["selected_bg"]};
    color: {PALETTE["selected_text"]};
    border-color: {PALETTE["primary"]};
    font-weight: 700;
}}

QLineEdit, QTextEdit, QComboBox {{
    background: {PALETTE["panel_bg"]};
    border: 1px solid {PALETTE["border_light"]};
    border-radius: 2px;
    padding: 3px 5px;
    color: {PALETTE["text"]};
    selection-background-color: {PALETTE["selected_bg"]};
    selection-color: {PALETTE["text"]};
}}
QLineEdit:hover, QTextEdit:hover, QComboBox:hover {{
    border-color: {PALETTE["border"]};
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border-color: {PALETTE["primary"]};
}}
QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled {{
    color: {PALETTE["text_disabled"]};
    background: #f1f3f5;
    border-color: {PALETTE["border_light"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QCheckBox {{
    spacing: 6px;
    color: {PALETTE["text_secondary"]};
}}
QCheckBox:disabled {{
    color: {PALETTE["text_disabled"]};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid #94a3b8;
    border-radius: 2px;
    background: #ffffff;
}}
QCheckBox::indicator:hover {{
    border-color: {PALETTE["primary"]};
}}
QCheckBox::indicator:checked {{
    border-color: {PALETTE["primary"]};
    background: {PALETTE["primary"]};
    image: url("{CHECK_ICON_PATH}");
}}
QCheckBox::indicator:checked:hover {{
    background: {PALETTE["primary_hover"]};
    border-color: {PALETTE["primary_hover"]};
}}
QCheckBox::indicator:disabled {{
    border-color: {PALETTE["border"]};
    background: #f1f3f5;
}}
QWidget#fp16Box {{
    background: {PALETTE["panel_subtle"]};
    border: 1px solid {PALETTE["border_light"]};
    border-radius: 2px;
}}

QWidget#emptyState {{
    background: {PALETTE["canvas_bg"]};
    color: #d1d5db;
}}
QLabel#emptyTitle {{
    color: #f3f4f6;
    font-size: 11pt;
    font-weight: 700;
}}
QLabel#emptySubtitle {{
    color: #cbd5e1;
}}
QPushButton#emptyPrimary, QPushButton#emptySecondary {{
    padding: 4px 9px;
    min-height: 18px;
}}
QPushButton#emptyPrimary {{
    background: #172033;
    border-color: #334155;
    color: #e5e7eb;
}}
QPushButton#emptyPrimary:hover {{
    background: #1e293b;
    border-color: {PALETTE["primary"]};
}}
QPushButton#emptySecondary {{
    background: #111827;
    border-color: #334155;
    color: #d1d5db;
}}
QPushButton#emptySecondary:hover {{
    background: #1f2937;
    border-color: #475569;
}}

QListWidget {{
    border: 1px solid {PALETTE["border_light"]};
    background: {PALETTE["panel_bg"]};
    border-radius: 2px;
    outline: none;
}}
QListWidget::item {{
    border-bottom: 1px solid {PALETTE["border_light"]};
    padding: 4px 6px;
    color: {PALETTE["text_secondary"]};
}}
QListWidget::item:selected {{
    background: {PALETTE["selected_bg"]};
    color: {PALETTE["selected_text"]};
    border-left: 2px solid {PALETTE["primary"]};
}}

QTableWidget {{
    background: {PALETTE["panel_bg"]};
    border: 1px solid {PALETTE["border_light"]};
    border-radius: 2px;
    gridline-color: {PALETTE["border_light"]};
    selection-background-color: {PALETTE["selected_bg"]};
    selection-color: {PALETTE["text"]};
}}
QHeaderView::section {{
    background: {PALETTE["panel_subtle"]};
    border: none;
    border-bottom: 1px solid {PALETTE["border_light"]};
    padding: 3px 5px;
    font-weight: 700;
    color: {PALETTE["text_secondary"]};
}}

QTabWidget::pane {{
    border: 1px solid {PALETTE["border_light"]};
    background: {PALETTE["panel_bg"]};
}}
QTabBar::tab {{
    background: #eef2f7;
    color: {PALETTE["text_secondary"]};
    padding: 4px 9px;
    border: 1px solid {PALETTE["border_light"]};
    border-bottom: none;
    margin-right: 1px;
}}
QTabBar::tab:hover {{
    background: {PALETTE["panel_subtle"]};
    color: {PALETTE["text"]};
}}
QTabBar::tab:selected {{
    background: {PALETTE["panel_bg"]};
    color: {PALETTE["text"]};
    font-weight: 700;
    border-top: 2px solid {PALETTE["primary"]};
}}

QProgressBar {{
    background: {PALETTE["panel_bg"]};
    border: 1px solid {PALETTE["border_light"]};
    border-radius: 2px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {PALETTE["primary"]};
}}

QStatusBar {{
    background: {PALETTE["panel_bg"]};
    border-top: 1px solid {PALETTE["border_light"]};
    color: {PALETTE["text_secondary"]};
}}
"""
