from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from sam3_auto_annotator.annotation.segmentation import (
    has_valid_segmentation,
    polygon_xyn_to_pixels,
)


MIN_BOX_SIZE = 2.0
HANDLE_SIZE = 8.0
HANDLE_MARGIN = 4.0
MASK_Z = 2
POLYGON_Z = 5
BBOX_Z = 10
HANDLE_Z = 20

BOX_COLORS = [
    QColor("#2563eb"),
    QColor("#16a34a"),
    QColor("#dc2626"),
    QColor("#9333ea"),
    QColor("#ea580c"),
    QColor("#0891b2"),
]


class AnnotationRectItem(QGraphicsRectItem):
    HANDLE_CURSORS = {
        "top_left": Qt.SizeFDiagCursor,
        "bottom_right": Qt.SizeFDiagCursor,
        "top_right": Qt.SizeBDiagCursor,
        "bottom_left": Qt.SizeBDiagCursor,
        "top": Qt.SizeVerCursor,
        "bottom": Qt.SizeVerCursor,
        "left": Qt.SizeHorCursor,
        "right": Qt.SizeHorCursor,
    }

    def __init__(self, annotation, image_rect, changed_callback=None):
        x1, y1, x2, y2 = annotation.box_xyxy
        super().__init__(0, 0, x2 - x1, y2 - y1)
        self.annotation_id = annotation.id
        self.class_id = annotation.class_id
        self._image_rect = QRectF(image_rect)
        self._changed_callback = changed_callback
        self._active_handle = None
        self._moving = False
        self._press_scene_pos = QPointF()
        self._press_scene_rect = QRectF()
        self.setPos(x1, y1)
        self.setZValue(BBOX_Z)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.apply_style(False)

    def apply_style(self, selected):
        color = BOX_COLORS[self.class_id % len(BOX_COLORS)]
        pen = QPen(QColor("#facc15") if selected else color, 3 if selected else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(28 if not selected else 42)
        self.setBrush(fill)

    def boundingRect(self):
        return super().boundingRect().adjusted(-HANDLE_MARGIN, -HANDLE_MARGIN, HANDLE_MARGIN, HANDLE_MARGIN)

    def paint(self, painter, option, widget=None):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())

        if not self.isSelected():
            return

        handle_pen = QPen(QColor("#ffffff"), 1)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        painter.setBrush(QColor("#2563eb"))
        for handle_rect in self._handle_rects().values():
            painter.drawRect(handle_rect)

    def hoverMoveEvent(self, event):
        handle = self._handle_at(event.pos())
        self.setCursor(self.HANDLE_CURSORS.get(handle, Qt.SizeAllCursor))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        self.setSelected(True)
        self._active_handle = self._handle_at(event.pos()) if self.isSelected() else None
        self._moving = self._active_handle is None and self.rect().contains(event.pos())
        self._press_scene_pos = event.scenePos()
        self._press_scene_rect = self._scene_rect()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._active_handle:
            self._resize_from_handle(event.scenePos())
            event.accept()
            return
        if self._moving:
            self._move_to(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._active_handle or self._moving:
            self._active_handle = None
            self._moving = False
            self._emit_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _handle_rects(self):
        rect = self.rect()
        half = HANDLE_SIZE / 2.0
        center_x = rect.center().x()
        center_y = rect.center().y()
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        centers = {
            "top_left": QPointF(left, top),
            "top": QPointF(center_x, top),
            "top_right": QPointF(right, top),
            "right": QPointF(right, center_y),
            "bottom_right": QPointF(right, bottom),
            "bottom": QPointF(center_x, bottom),
            "bottom_left": QPointF(left, bottom),
            "left": QPointF(left, center_y),
        }
        return {
            name: QRectF(center.x() - half, center.y() - half, HANDLE_SIZE, HANDLE_SIZE)
            for name, center in centers.items()
        }

    def _handle_at(self, point):
        for name, rect in self._handle_rects().items():
            if rect.contains(point):
                return name
        return None

    def _scene_rect(self):
        rect = self.rect()
        top_left = self.mapToScene(rect.topLeft())
        bottom_right = self.mapToScene(rect.bottomRight())
        return QRectF(top_left, bottom_right).normalized()

    def _apply_scene_rect(self, scene_rect):
        rect = scene_rect.normalized().intersected(self._image_rect)
        if rect.width() < MIN_BOX_SIZE or rect.height() < MIN_BOX_SIZE:
            return
        self.prepareGeometryChange()
        self.setPos(rect.left(), rect.top())
        self.setRect(0, 0, rect.width(), rect.height())
        self.update()

    def _resize_from_handle(self, scene_pos):
        point = self._clamp_scene_point(scene_pos)
        rect = QRectF(self._press_scene_rect)
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        if "left" in self._active_handle:
            x1 = min(point.x(), x2 - MIN_BOX_SIZE)
            x1 = max(x1, self._image_rect.left())
        if "right" in self._active_handle:
            x2 = max(point.x(), x1 + MIN_BOX_SIZE)
            x2 = min(x2, self._image_rect.right())
        if "top" in self._active_handle:
            y1 = min(point.y(), y2 - MIN_BOX_SIZE)
            y1 = max(y1, self._image_rect.top())
        if "bottom" in self._active_handle:
            y2 = max(point.y(), y1 + MIN_BOX_SIZE)
            y2 = min(y2, self._image_rect.bottom())

        self._apply_scene_rect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))

    def _move_to(self, scene_pos):
        delta = scene_pos - self._press_scene_pos
        rect = QRectF(self._press_scene_rect)
        rect.translate(delta)

        if rect.left() < self._image_rect.left():
            rect.moveLeft(self._image_rect.left())
        if rect.top() < self._image_rect.top():
            rect.moveTop(self._image_rect.top())
        if rect.right() > self._image_rect.right():
            rect.moveRight(self._image_rect.right())
        if rect.bottom() > self._image_rect.bottom():
            rect.moveBottom(self._image_rect.bottom())

        self._apply_scene_rect(rect)

    def _clamp_scene_point(self, point):
        return QPointF(
            min(max(point.x(), self._image_rect.left()), self._image_rect.right()),
            min(max(point.y(), self._image_rect.top()), self._image_rect.bottom()),
        )

    def _emit_changed(self):
        if self._changed_callback is None:
            return
        rect = self._scene_rect().intersected(self._image_rect)
        self._changed_callback(
            self.annotation_id,
            (rect.left(), rect.top(), rect.right(), rect.bottom()),
        )


class ImageCanvas(QGraphicsView):
    box_drawn = Signal(tuple)
    annotation_selected = Signal(object)
    annotation_changed = Signal(str, tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QColor("#111827"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)

        self._pixmap_item = None
        self._image_rect = QRectF()
        self._items_by_id = {}
        self._mask_items_by_id = {}
        self._polygon_items_by_id = {}
        self._annotations = []
        self._show_boxes = True
        self._show_masks = True
        self._show_polygons = False
        self._draw_mode = False
        self._drawing = False
        self._draw_start = QPointF()
        self._draft_item = None

        self._scene.selectionChanged.connect(self._on_selection_changed)

    def set_draw_mode(self, enabled):
        self._draw_mode = bool(enabled)
        self.viewport().setCursor(Qt.CrossCursor if self._draw_mode else Qt.ArrowCursor)

    def load_image(self, image_path):
        image = QImage(str(image_path))
        if image.isNull():
            raise ValueError(f"Could not load image: {image_path}")

        self._scene.clear()
        self._items_by_id = {}
        self._mask_items_by_id = {}
        self._polygon_items_by_id = {}
        self._annotations = []
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)
        self._image_rect = QRectF(0, 0, image.width(), image.height())
        self._scene.setSceneRect(self._image_rect)
        self.fit_to_window()
        return image.width(), image.height()

    def set_annotations(self, annotations):
        self._annotations = list(annotations)
        for item in list(self._items_by_id.values()):
            self._scene.removeItem(item)
        for item in list(self._mask_items_by_id.values()):
            self._scene.removeItem(item)
        for item in list(self._polygon_items_by_id.values()):
            self._scene.removeItem(item)
        self._items_by_id = {}
        self._mask_items_by_id = {}
        self._polygon_items_by_id = {}

        for annotation in self._annotations:
            if annotation.deleted:
                continue
            self._add_segmentation_items(annotation)
            item = AnnotationRectItem(
                annotation,
                image_rect=self._image_rect,
                changed_callback=self._emit_annotation_changed,
            )
            self._items_by_id[annotation.id] = item
            item.setVisible(self._show_boxes)
            self._scene.addItem(item)

    def set_overlay_visibility(self, show_boxes=None, show_masks=None, show_polygons=None):
        if show_boxes is not None:
            self._show_boxes = bool(show_boxes)
        if show_masks is not None:
            self._show_masks = bool(show_masks)
        if show_polygons is not None:
            self._show_polygons = bool(show_polygons)

        for item in self._items_by_id.values():
            item.setVisible(self._show_boxes)
        for item in self._mask_items_by_id.values():
            item.setVisible(self._show_masks)
        for item in self._polygon_items_by_id.values():
            item.setVisible(self._show_polygons)

    def selected_annotation_id(self):
        for item in self._scene.selectedItems():
            if isinstance(item, AnnotationRectItem):
                return item.annotation_id
        return None

    def select_annotation(self, annotation_id):
        self._scene.blockSignals(True)
        for item in self._items_by_id.values():
            item.setSelected(item.annotation_id == annotation_id)
            item.apply_style(item.isSelected())
        self._scene.blockSignals(False)
        self._refresh_segmentation_styles(annotation_id)
        self.annotation_selected.emit(annotation_id)

    def remove_annotation(self, annotation_id):
        item = self._items_by_id.pop(annotation_id, None)
        if item is not None:
            self._scene.removeItem(item)
        mask_item = self._mask_items_by_id.pop(annotation_id, None)
        if mask_item is not None:
            self._scene.removeItem(mask_item)
        polygon_item = self._polygon_items_by_id.pop(annotation_id, None)
        if polygon_item is not None:
            self._scene.removeItem(polygon_item)

    def update_annotation_box(self, annotation):
        item = self._items_by_id.get(annotation.id)
        if item is not None:
            x1, y1, x2, y2 = annotation.box_xyxy
            item.setRect(0, 0, x2 - x1, y2 - y1)
            item.setPos(x1, y1)
        self._sync_segmentation_items(annotation)

    def fit_to_window(self):
        if not self._image_rect.isNull():
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton and self._pixmap_item is not None:
            point = self.mapToScene(event.position().toPoint())
            if self._image_rect.contains(point):
                self._drawing = True
                self._draw_start = point
                self._draft_item = QGraphicsRectItem(QRectF(point, point))
                pen = QPen(QColor("#facc15"), 2, Qt.DashLine)
                pen.setCosmetic(True)
                self._draft_item.setPen(pen)
                self._draft_item.setBrush(QColor(250, 204, 21, 35))
                self._draft_item.setZValue(HANDLE_Z)
                self._scene.addItem(self._draft_item)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._draft_item is not None:
            point = self._clamp_point(self.mapToScene(event.position().toPoint()))
            rect = QRectF(self._draw_start, point).normalized()
            self._draft_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing and self._draft_item is not None:
            rect = self._draft_item.rect().normalized()
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
            self._drawing = False
            if rect.width() >= 3 and rect.height() >= 3:
                self.box_drawn.emit((rect.left(), rect.top(), rect.right(), rect.bottom()))
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self._emit_changed_boxes()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.fit_to_window()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None and self.transform().m11() == 1.0:
            self.fit_to_window()

    def _add_segmentation_items(self, annotation):
        if not has_valid_segmentation(annotation) or self._image_rect.isNull():
            return

        pixel_points = polygon_xyn_to_pixels(
            annotation.polygon_xyn,
            self._image_rect.width(),
            self._image_rect.height(),
        )
        polygon = QPolygonF([QPointF(x, y) for x, y in pixel_points])
        if polygon.size() < 3:
            return

        path = QPainterPath()
        path.addPolygon(polygon)
        path.closeSubpath()

        mask_item = QGraphicsPathItem(path)
        mask_item.setZValue(MASK_Z)
        self._configure_overlay_item(mask_item)
        mask_item.setVisible(self._show_masks)
        self._mask_items_by_id[annotation.id] = mask_item
        self._scene.addItem(mask_item)

        polygon_item = QGraphicsPathItem(path)
        polygon_item.setZValue(POLYGON_Z)
        self._configure_overlay_item(polygon_item)
        polygon_item.setBrush(QBrush(Qt.NoBrush))
        polygon_item.setVisible(self._show_polygons)
        self._polygon_items_by_id[annotation.id] = polygon_item
        self._scene.addItem(polygon_item)
        self._apply_segmentation_style(annotation, mask_item, polygon_item)

    def _configure_overlay_item(self, item):
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        item.setCacheMode(QGraphicsItem.NoCache)

    def _apply_segmentation_style(self, annotation, mask_item, polygon_item, selected=False):
        fill = QColor("#0891b2")
        fill.setAlpha(115 if selected else 95)
        mask_item.setBrush(QBrush(fill))
        mask_item.setPen(QPen(Qt.NoPen))

        outline = QColor("#facc15" if selected else "#22d3ee")
        outline.setAlpha(255 if selected else 235)
        pen = QPen(outline, 2.4 if selected else 1.8)
        pen.setCosmetic(True)
        polygon_item.setPen(pen)

    def _refresh_segmentation_styles(self, selected_id=None):
        annotations_by_id = {annotation.id: annotation for annotation in self._annotations}
        for annotation_id, polygon_item in self._polygon_items_by_id.items():
            annotation = annotations_by_id.get(annotation_id)
            mask_item = self._mask_items_by_id.get(annotation_id)
            if annotation is None or mask_item is None:
                continue
            self._apply_segmentation_style(
                annotation,
                mask_item,
                polygon_item,
                selected=annotation_id == selected_id,
            )

    def _sync_segmentation_items(self, annotation):
        mask_item = self._mask_items_by_id.pop(annotation.id, None)
        if mask_item is not None:
            self._scene.removeItem(mask_item)
        polygon_item = self._polygon_items_by_id.pop(annotation.id, None)
        if polygon_item is not None:
            self._scene.removeItem(polygon_item)
        self._add_segmentation_items(annotation)

    def _on_selection_changed(self):
        selected_id = self.selected_annotation_id()
        for item in self._items_by_id.values():
            item.apply_style(item.annotation_id == selected_id)
        self._refresh_segmentation_styles(selected_id)
        self.annotation_selected.emit(selected_id)

    def _clamp_point(self, point):
        return QPointF(
            min(max(point.x(), self._image_rect.left()), self._image_rect.right()),
            min(max(point.y(), self._image_rect.top()), self._image_rect.bottom()),
        )

    def _emit_changed_boxes(self):
        for annotation_id, item in self._items_by_id.items():
            rect = item._scene_rect().intersected(self._image_rect)
            if rect.width() <= 0 or rect.height() <= 0:
                continue
            item.setPos(rect.left(), rect.top())
            item.setRect(0, 0, rect.width(), rect.height())
            self._emit_annotation_changed(
                annotation_id,
                (rect.left(), rect.top(), rect.right(), rect.bottom()),
            )

    def _emit_annotation_changed(self, annotation_id, box_xyxy):
        self.annotation_changed.emit(annotation_id, box_xyxy)
