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
    button.setSizePolicy(
        QSizePolicy.Expanding if stretch else QSizePolicy.Fixed,
        QSizePolicy.Fixed,
    )
    role = action.property("role")
    if object_name:
        button.setObjectName(object_name)
    elif role:
        button.setObjectName(f"{role}Button")
    return button
