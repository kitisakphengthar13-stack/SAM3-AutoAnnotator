from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF

from domain.segmentation import (
    has_valid_segmentation,
    polygon_xyn_to_pixels,
)
from gui.widgets.image_canvas import BOX_COLORS


@dataclass(frozen=True)
class OverlayOptions:
    boxes: bool = True
    masks: bool = True
    polygons: bool = False


def render_annotation_preview(image_path, annotations, output_path, options):
    """Render a reviewed preview without reading or mutating UI widgets."""
    qimage = QImage(str(image_path))
    if qimage.isNull():
        raise ValueError(f"Could not load image: {image_path}")

    painter = QPainter(qimage)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        if options.masks:
            _draw_masks(painter, qimage, annotations)
        if options.polygons:
            _draw_polygons(painter, qimage, annotations)
        if options.boxes:
            _draw_boxes(painter, qimage, annotations)
    finally:
        painter.end()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not qimage.save(str(output_path)):
        raise OSError(f"Could not save preview image: {output_path}")
    return output_path


def _polygon(annotation, image):
    return QPolygonF(
        [
            QPointF(x, y)
            for x, y in polygon_xyn_to_pixels(
                annotation.polygon_xyn,
                image.width(),
                image.height(),
            )
        ]
    )


def _draw_masks(painter, image, annotations):
    for annotation in annotations:
        if not has_valid_segmentation(annotation):
            continue
        polygon = _polygon(annotation, image)
        if polygon.size() < 3:
            continue
        color = QColor("#0891b2")
        color.setAlpha(95)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(polygon)


def _draw_polygons(painter, image, annotations):
    for annotation in annotations:
        if not has_valid_segmentation(annotation):
            continue
        polygon = _polygon(annotation, image)
        if polygon.size() < 3:
            continue
        color = QColor("#22d3ee")
        color.setAlpha(235)
        painter.setPen(QPen(color, max(2, int(image.width() / 760))))
        painter.setBrush(QBrush(Qt.NoBrush))
        painter.drawPolygon(polygon)


def _draw_boxes(painter, image, annotations):
    for annotation in annotations:
        color = BOX_COLORS[annotation.class_id % len(BOX_COLORS)]
        painter.setPen(QPen(color, max(2, int(image.width() / 640))))
        painter.setBrush(QBrush(Qt.NoBrush))
        x1, y1, x2, y2 = annotation.box_xyxy
        painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        painter.drawText(int(x1) + 4, max(14, int(y1) - 4), annotation.class_name)
