from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)
from shiboken6 import isValid

from domain.segmentation import (
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
    QColor("#78a9ff"),
    QColor("#63d8bb"),
    QColor("#ff8c95"),
    QColor("#c4a7fa"),
    QColor("#f4c078"),
    QColor("#78d5ed"),
]


class ResizeHandleItem(QGraphicsRectItem):
    """Constant device-size handle that delegates resizing to its owner box."""

    def __init__(self, name, owner):
        half = HANDLE_SIZE / 2.0
        super().__init__(-half, -half, HANDLE_SIZE, HANDLE_SIZE, owner)
        self.name = name
        self.owner = owner
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setCursor(owner.HANDLE_CURSORS[name])
        self.setPen(QPen(QColor("#ffffff"), 1))
        self.setBrush(QColor("#2563eb"))
        self.setZValue(HANDLE_Z)
        self.setVisible(False)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self.owner.setSelected(True)
        self.owner._active_handle = self.name
        self.owner._press_scene_rect = self.owner._scene_rect()
        event.accept()

    def mouseMoveEvent(self, event):
        self.owner._resize_from_handle(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        self.owner._active_handle = None
        self.owner._emit_changed()
        event.accept()


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
        self._handle_items = {
            name: ResizeHandleItem(name, self) for name in self.HANDLE_CURSORS
        }
        self._position_handle_items()
        self.apply_style(False)

    def setRect(self, *args):
        super().setRect(*args)
        if hasattr(self, "_handle_items"):
            self._position_handle_items()

    def apply_style(self, selected):
        color = BOX_COLORS[self.class_id % len(BOX_COLORS)]
        pen = QPen(QColor("#ffffff") if selected else color, 3 if selected else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(10 if not selected else 18)
        self.setBrush(fill)
        for handle in self._handle_items.values():
            handle.setVisible(bool(selected))

    def boundingRect(self):
        return (
            super()
            .boundingRect()
            .adjusted(-HANDLE_MARGIN, -HANDLE_MARGIN, HANDLE_MARGIN, HANDLE_MARGIN)
        )

    def paint(self, painter, option, widget=None):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())

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
        self._active_handle = (
            self._handle_at(event.pos()) if self.isSelected() else None
        )
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

    def _position_handle_items(self):
        rect = self.rect()
        centers = {
            "top_left": rect.topLeft(),
            "top": QPointF(rect.center().x(), rect.top()),
            "top_right": rect.topRight(),
            "right": QPointF(rect.right(), rect.center().y()),
            "bottom_right": rect.bottomRight(),
            "bottom": QPointF(rect.center().x(), rect.bottom()),
            "bottom_left": rect.bottomLeft(),
            "left": QPointF(rect.left(), rect.center().y()),
        }
        for name, center in centers.items():
            self._handle_items[name].setPos(center)

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
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
        self.setBackgroundBrush(QColor("#111827"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)

        self._pixmap_item = None
        self._image_rect = QRectF()
        self._items_by_id = {}
        self._mask_items_by_id = {}
        self._polygon_items_by_id = {}
        self._render_signatures = {}
        self._annotations = []
        self._selected_annotation_id = None
        self._show_boxes = True
        self._show_masks = True
        self._show_polygons = False
        self._draw_mode = False
        self._drawing = False
        self._draw_start = QPointF()
        self._draft_item = None
        self._auto_fit = True

        self._scene.selectionChanged.connect(self._on_selection_changed)

    def set_draw_mode(self, enabled):
        self._draw_mode = bool(enabled)
        self.viewport().setCursor(Qt.CrossCursor if self._draw_mode else Qt.ArrowCursor)

    def clear_image(self):
        """Remove all image-owned graphics so stale content can never survive."""
        self._scene.blockSignals(True)
        try:
            self._scene.clear()
            self._pixmap_item = None
            self._items_by_id.clear()
            self._mask_items_by_id.clear()
            self._polygon_items_by_id.clear()
            self._render_signatures.clear()
        finally:
            self._scene.blockSignals(False)
        self._annotations = []
        self._selected_annotation_id = None
        self._image_rect = QRectF()
        self._drawing = False
        self._draft_item = None
        self._scene.setSceneRect(QRectF())
        self.resetTransform()

    def load_image(self, image_path):
        self.clear_image()
        image = QImage(str(image_path))
        if image.isNull():
            raise ValueError(f"Could not load image: {image_path}")

        pixmap = QPixmap.fromImage(image)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)
        self._image_rect = QRectF(0, 0, image.width(), image.height())
        self._scene.setSceneRect(self._image_rect)
        self.fit_to_window()
        return image.width(), image.height()

    def set_annotations(self, annotations):
        previous_selected_id = self._selected_annotation_id
        self._annotations = list(annotations)
        active_by_id = {
            annotation.id: annotation
            for annotation in self._annotations
            if not annotation.deleted
        }

        self._scene.blockSignals(True)
        try:
            self._prune_item_registries()
            for annotation_id in set(self._items_by_id) - set(active_by_id):
                self.remove_annotation(annotation_id)

            for annotation in active_by_id.values():
                signature = self._render_signature(annotation)
                item = self._items_by_id.get(annotation.id)
                overlays_expected = has_valid_segmentation(annotation)
                overlays_live = self._overlays_live(annotation.id)
                if not self._is_live_item(item):
                    if item is not None:
                        self._items_by_id.pop(annotation.id, None)
                    self._add_annotation_graphics(annotation)
                    self._render_signatures[annotation.id] = signature
                    continue
                if (
                    self._render_signatures.get(annotation.id) != signature
                    or overlays_expected != overlays_live
                ):
                    self._update_annotation_graphics(annotation)
                    self._render_signatures[annotation.id] = signature

            self._selected_annotation_id = (
                previous_selected_id
                if previous_selected_id in self._items_by_id
                else None
            )
            selected_item = self._items_by_id.get(self._selected_annotation_id)
            if self._is_live_item(selected_item):
                selected_item.setSelected(True)
                selected_item.apply_style(True)
                self._refresh_annotation_segmentation_style(
                    self._selected_annotation_id,
                    selected=True,
                )
        finally:
            self._scene.blockSignals(False)

    def set_overlay_visibility(
        self, show_boxes=None, show_masks=None, show_polygons=None
    ):
        if show_boxes is not None:
            self._show_boxes = bool(show_boxes)
        if show_masks is not None:
            self._show_masks = bool(show_masks)
        if show_polygons is not None:
            self._show_polygons = bool(show_polygons)

        for item in self._items_by_id.values():
            if self._is_live_item(item):
                item.setVisible(self._show_boxes)
        for item in self._mask_items_by_id.values():
            if self._is_live_item(item):
                item.setVisible(self._show_masks)
        for item in self._polygon_items_by_id.values():
            if self._is_live_item(item):
                item.setVisible(self._show_polygons)

    def selected_annotation_id(self, preserve_stored=True):
        self._prune_item_registries()
        for item in self._scene.selectedItems():
            if isinstance(item, AnnotationRectItem) and self._is_live_item(item):
                self._selected_annotation_id = item.annotation_id
                return item.annotation_id
        if preserve_stored and self._selected_annotation_id in self._items_by_id:
            return self._selected_annotation_id
        self._selected_annotation_id = None
        return None

    def select_annotation(self, annotation_id):
        self._scene.blockSignals(True)
        try:
            self._prune_item_registries()
            self._selected_annotation_id = (
                annotation_id if annotation_id in self._items_by_id else None
            )
            self._apply_selection_to_items(self._selected_annotation_id)
        finally:
            self._scene.blockSignals(False)
        self._refresh_segmentation_styles(self._selected_annotation_id)
        self.annotation_selected.emit(self._selected_annotation_id)

    def remove_annotation(self, annotation_id):
        item = self._items_by_id.pop(annotation_id, None)
        self._remove_scene_item(item)
        mask_item = self._mask_items_by_id.pop(annotation_id, None)
        self._remove_scene_item(mask_item)
        polygon_item = self._polygon_items_by_id.pop(annotation_id, None)
        self._remove_scene_item(polygon_item)
        self._render_signatures.pop(annotation_id, None)
        if self._selected_annotation_id == annotation_id:
            self._selected_annotation_id = None

    def update_annotation_box(self, annotation):
        if annotation.deleted:
            self.remove_annotation(annotation.id)
            return
        item = self._items_by_id.get(annotation.id)
        if self._is_live_item(item):
            self._update_annotation_graphics(annotation)
        else:
            if item is not None:
                self._items_by_id.pop(annotation.id, None)
            self._add_annotation_graphics(annotation)
        self._render_signatures[annotation.id] = self._render_signature(annotation)

    def fit_to_window(self):
        if not self._image_rect.isNull():
            self._auto_fit = True
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        delta = event.angleDelta().y()
        if not delta:
            event.accept()
            return
        self._auto_fit = False
        factor = 1.15 if delta > 0 else 1 / 1.15
        current = self.transform().m11()
        target = max(0.05, min(20.0, current * factor))
        self.scale(target / current, target / current)
        event.accept()

    def mousePressEvent(self, event):
        if (
            self._draw_mode
            and event.button() == Qt.LeftButton
            and self._pixmap_item is not None
        ):
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
                self.box_drawn.emit(
                    (rect.left(), rect.top(), rect.right(), rect.bottom())
                )
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None and self._auto_fit:
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def _render_signature(self, annotation):
        segmentation_valid = has_valid_segmentation(annotation)
        polygon = (
            tuple((float(x), float(y)) for x, y in annotation.polygon_xyn or [])
            if segmentation_valid
            else ()
        )
        return (
            int(annotation.class_id),
            tuple(float(value) for value in annotation.box_xyxy),
            segmentation_valid,
            polygon,
        )

    def _overlays_live(self, annotation_id):
        return self._is_live_item(
            self._mask_items_by_id.get(annotation_id)
        ) and self._is_live_item(self._polygon_items_by_id.get(annotation_id))

    def _add_annotation_graphics(self, annotation):
        self._add_segmentation_items(annotation)
        item = AnnotationRectItem(
            annotation,
            image_rect=self._image_rect,
            changed_callback=self._emit_annotation_changed,
        )
        selected = annotation.id == self._selected_annotation_id
        item.setSelected(selected)
        item.apply_style(selected)
        item.setVisible(self._show_boxes)
        self._items_by_id[annotation.id] = item
        self._scene.addItem(item)

    def _update_annotation_graphics(self, annotation):
        item = self._items_by_id.get(annotation.id)
        if not self._is_live_item(item):
            return
        x1, y1, x2, y2 = annotation.box_xyxy
        item.class_id = annotation.class_id
        item.setRect(0, 0, x2 - x1, y2 - y1)
        item.setPos(x1, y1)
        item.setVisible(self._show_boxes)
        item.apply_style(annotation.id == self._selected_annotation_id)
        self._sync_segmentation_items(annotation)

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
        self._apply_segmentation_style(
            annotation,
            mask_item,
            polygon_item,
            selected=annotation.id == self._selected_annotation_id,
        )

    def _is_live_item(self, item):
        return item is not None and isValid(item)

    def _remove_scene_item(self, item):
        if self._is_live_item(item):
            self._scene.removeItem(item)

    def _remove_registry_items(self, registry):
        for item in list(registry.values()):
            self._remove_scene_item(item)

    def _prune_registry(self, registry):
        stale_ids = [
            annotation_id
            for annotation_id, item in registry.items()
            if not self._is_live_item(item)
        ]
        for annotation_id in stale_ids:
            registry.pop(annotation_id, None)

    def _prune_item_registries(self):
        self._prune_registry(self._items_by_id)
        self._prune_registry(self._mask_items_by_id)
        self._prune_registry(self._polygon_items_by_id)
        for annotation_id in list(self._render_signatures):
            if annotation_id not in self._items_by_id:
                self._render_signatures.pop(annotation_id, None)
        if self._selected_annotation_id not in self._items_by_id:
            self._selected_annotation_id = None

    def _apply_selection_to_items(self, selected_id):
        for item in list(self._items_by_id.values()):
            if not self._is_live_item(item):
                continue
            selected = item.annotation_id == selected_id
            item.setSelected(selected)
            item.apply_style(selected)

    def _configure_overlay_item(self, item):
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        item.setCacheMode(QGraphicsItem.NoCache)

    def _apply_segmentation_style(
        self, annotation, mask_item, polygon_item, selected=False
    ):
        class_color = BOX_COLORS[annotation.class_id % len(BOX_COLORS)]
        fill = QColor(class_color)
        fill.setAlpha(110 if selected else 82)
        mask_item.setBrush(QBrush(fill))
        mask_item.setPen(QPen(Qt.NoPen))

        outline = QColor("#facc15" if selected else class_color)
        outline.setAlpha(255 if selected else 235)
        pen = QPen(outline, 2.4 if selected else 1.8)
        pen.setCosmetic(True)
        polygon_item.setPen(pen)

    def _refresh_annotation_segmentation_style(self, annotation_id, selected=False):
        annotation = next(
            (item for item in self._annotations if item.id == annotation_id),
            None,
        )
        mask_item = self._mask_items_by_id.get(annotation_id)
        polygon_item = self._polygon_items_by_id.get(annotation_id)
        if (
            annotation is None
            or not self._is_live_item(mask_item)
            or not self._is_live_item(polygon_item)
        ):
            return
        self._apply_segmentation_style(
            annotation,
            mask_item,
            polygon_item,
            selected=selected,
        )

    def _refresh_segmentation_styles(self, selected_id=None):
        self._prune_item_registries()
        annotations_by_id = {
            annotation.id: annotation for annotation in self._annotations
        }
        for annotation_id, polygon_item in list(self._polygon_items_by_id.items()):
            annotation = annotations_by_id.get(annotation_id)
            mask_item = self._mask_items_by_id.get(annotation_id)
            if (
                annotation is None
                or not self._is_live_item(mask_item)
                or not self._is_live_item(polygon_item)
            ):
                continue
            self._apply_segmentation_style(
                annotation,
                mask_item,
                polygon_item,
                selected=annotation_id == selected_id,
            )

    def _sync_segmentation_items(self, annotation):
        mask_item = self._mask_items_by_id.pop(annotation.id, None)
        self._remove_scene_item(mask_item)
        polygon_item = self._polygon_items_by_id.pop(annotation.id, None)
        self._remove_scene_item(polygon_item)
        self._add_segmentation_items(annotation)

    def _on_selection_changed(self):
        selected_id = self.selected_annotation_id(preserve_stored=False)
        self._apply_selection_to_items(selected_id)
        self._refresh_segmentation_styles(selected_id)
        self.annotation_selected.emit(selected_id)

    def _clamp_point(self, point):
        return QPointF(
            min(max(point.x(), self._image_rect.left()), self._image_rect.right()),
            min(max(point.y(), self._image_rect.top()), self._image_rect.bottom()),
        )

    def _emit_annotation_changed(self, annotation_id, box_xyxy):
        self.annotation_changed.emit(annotation_id, box_xyxy)
