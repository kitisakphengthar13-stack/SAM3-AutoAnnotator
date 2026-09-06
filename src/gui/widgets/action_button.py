from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QSizePolicy, QToolButton


def action_button(
    action: QAction,
    object_name: str | None = None,
    *,
    icon_only: bool = False,
    stretch: bool = False,
) -> QToolButton:
    """Create a tool button backed by a shared application action."""

    button = QToolButton()
    button.setDefaultAction(action)
    button.setToolButtonStyle(
        Qt.ToolButtonIconOnly if icon_only else Qt.ToolButtonTextBesideIcon
    )
    button.setAutoRaise(icon_only)
    button.setProperty("iconOnly", icon_only)
    button.setAccessibleName(action.text().replace("&", ""))
    button.setSizePolicy(
        QSizePolicy.Expanding if stretch else QSizePolicy.Fixed,
        QSizePolicy.Fixed,
    )
    role = action.property("role")
    if object_name:
        button.setObjectName(object_name)
    elif role:
        button.setObjectName(f"{role}Button")
    if button.objectName() in {"primaryButton", "exportButton"}:
        from gui.icons import icon

        def sync_icon():
            button.setIcon(icon(action.property("iconName"), "#09251e"))

        action.changed.connect(sync_icon)
        sync_icon()
    return button


def menu_button(text, icon_name, menu):
    """The full button opens the menu; no tiny split-button target."""
    from gui.icons import icon

    button = QToolButton()
    button.setText(text)
    button.setIcon(icon(icon_name))
    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setPopupMode(QToolButton.InstantPopup)
    button.setMenu(menu)
    button.setProperty("popup", True)
    button.setAccessibleName(text)
    return button
