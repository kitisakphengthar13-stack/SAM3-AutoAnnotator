"""Compact, readable projections of the existing project-owned models."""

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QStyle, QStyledItemDelegate
from gui.models.image_list_model import (
    IMAGE_INDEX_ROLE,
    IMAGE_NAME_ROLE,
    IMAGE_STATUS_ROLE,
    ANNOTATION_COUNT_ROLE,
)
from gui.models.annotation_table_model import ANNOTATION_ROLE
from gui.widgets.image_canvas import BOX_COLORS

STATUS_COLORS = {
    "not_predicted": "#8796aa",
    "predicted": "#85baff",
    "edited": "#f4c078",
    "reviewed": "#63d8bb",
    "no_detection": "#c4a7fa",
    "error": "#ff8c95",
}


def surface(painter, option):
    painter.save()
    selected = bool(option.state & QStyle.State_Selected)
    hovered = bool(option.state & QStyle.State_MouseOver)
    rect = option.rect.adjusted(0, 2, 0, -2)
    painter.setPen(Qt.NoPen)
    painter.setBrush(
        QColor("#203b3b" if selected else "#222d39" if hovered else "#171d25")
    )
    painter.drawRoundedRect(rect, 5, 5)
    if selected:
        painter.fillRect(
            QRect(rect.left(), rect.top() + 8, 3, rect.height() - 16), QColor("#63d8bb")
        )
    return rect


def line(painter, rect, text, color, bold=False, size=9):
    font = QFont(painter.font())
    font.setPointSize(size)
    font.setBold(bold)
    painter.setFont(font)
    painter.setPen(QColor(color))
    text = painter.fontMetrics().elidedText(
        str(text), Qt.ElideMiddle, max(0, rect.width())
    )
    painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)


class DatasetDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(190, 65)

    def paint(self, painter, option, index):
        rect = surface(painter, option)
        x, y = rect.left(), rect.top()
        tile = QRect(x + 10, y + 13, 32, 32)
        painter.setBrush(QColor("#293746"))
        painter.drawRoundedRect(tile, 5, 5)
        painter.setPen(QColor("#acb9c9"))
        font = QFont(option.font)
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            tile, Qt.AlignCenter, f"{int(index.data(IMAGE_INDEX_ROLE))+1:02d}"
        )
        width = rect.width() - 62
        line(
            painter,
            QRect(x + 52, y + 7, width, 23),
            index.data(IMAGE_NAME_ROLE),
            "#e4eaf2",
            True,
        )
        status = index.data(IMAGE_STATUS_ROLE)
        text = str(status).replace("_", " ").capitalize()
        count = index.data(ANNOTATION_COUNT_ROLE)
        line(
            painter,
            QRect(x + 52, y + 31, width, 20),
            f"{text} · {count} objects",
            STATUS_COLORS.get(status, "#8796aa"),
            size=8,
        )
        painter.restore()


class ObjectDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(230, 61)

    def paint(self, painter, option, index):
        annotation = index.data(ANNOTATION_ROLE)
        if annotation is None:
            return
        rect = surface(painter, option)
        x, y = rect.left(), rect.top()
        painter.setBrush(BOX_COLORS[annotation.class_id % len(BOX_COLORS)])
        painter.drawRoundedRect(QRect(x + 12, y + 15, 10, 10), 3, 3)
        line(
            painter,
            QRect(x + 32, y + 5, rect.width() - 100, 25),
            annotation.class_name,
            "#e4eaf2",
            True,
            10,
        )
        confidence = (
            "—" if annotation.confidence is None else f"{annotation.confidence:.0%}"
        )
        line(painter, QRect(rect.right() - 56, y + 5, 48, 25), confidence, "#acb9c9")
        from domain.segmentation import segmentation_status

        status = segmentation_status(annotation)
        state = {
            "valid": "Mask ready",
            "stale": "Mask needs refresh",
            "none": "Box only",
            "invalid": "Invalid mask",
        }.get(status, status)
        source = annotation.source.value.replace("_", " ").capitalize()
        line(
            painter,
            QRect(x + 32, y + 30, rect.width() - 42, 20),
            f"{source} · {state}",
            "#f4c078" if status in ("stale", "invalid") else "#8796aa",
            size=8,
        )
        painter.restore()
