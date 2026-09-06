import logging

from domain import Annotation, AnnotationSource


logger = logging.getLogger(__name__)


def tensor_item(value, default=None):
    try:
        return value.item()
    except AttributeError:
        return value if value is not None else default


def get_sequence_value(sequence, index, default=None):
    if sequence is None:
        return default
    try:
        return sequence[index]
    except (IndexError, TypeError):
        return default


def get_class_name(class_id, prompts):
    if 0 <= class_id < len(prompts):
        return prompts[class_id]
    return "unknown"


def result_image_size(result):
    shape = getattr(result, "orig_shape", None)
    if shape and len(shape) >= 2:
        height, width = shape[:2]
        return int(width), int(height)
    return None, None


def _xyxy_values(boxes, object_index):
    xyxy = getattr(boxes, "xyxy", None)
    values = get_sequence_value(xyxy, object_index)
    if values is None:
        return None
    return tuple(float(tensor_item(value)) for value in values[:4])


def _polygon_values(polygons, object_index):
    polygon = get_sequence_value(polygons, object_index)
    if polygon is None:
        return None
    return [[float(x), float(y)] for x, y in polygon]


def _confidence_values(boxes):
    confidences = getattr(boxes, "conf", None)
    if confidences is None:
        return []
    try:
        count = len(confidences)
    except TypeError:
        return []
    return [tensor_item(get_sequence_value(confidences, index)) for index in range(count)]


def _mask_debug_values(masks, object_index):
    data = getattr(masks, "data", None)
    mask = get_sequence_value(data, object_index)
    if mask is None:
        return None, None
    shape = tuple(int(value) for value in getattr(mask, "shape", ()) or ())
    try:
        area = float(mask.sum().item())
    except AttributeError:
        try:
            area = float(mask.sum())
        except Exception:
            area = None
    except Exception:
        area = None
    return shape, area


def _polygon_box_iou(polygon, requested_box, result):
    width, height = result_image_size(result)
    if width is None or height is None or not polygon:
        return None
    xs = [float(point[0]) * width for point in polygon]
    ys = [float(point[1]) * height for point in polygon]
    px1, py1, px2, py2 = min(xs), min(ys), max(xs), max(ys)
    rx1, ry1, rx2, ry2 = requested_box
    ix1, iy1 = max(px1, rx1), max(py1, ry1)
    ix2, iy2 = min(px2, rx2), min(py2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    polygon_box_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    request_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = polygon_box_area + request_area - intersection
    return intersection / union if union > 0 else 0.0


def best_box_prompt_segmentation(results, requested_box=None):
    candidates = []
    for result_index, result in enumerate(results or []):
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xyn", None) if masks is not None else None
        if polygons is None:
            continue
        boxes = getattr(result, "boxes", None)
        confidences = _confidence_values(boxes)
        try:
            polygon_count = len(polygons)
        except TypeError:
            continue
        for polygon_index in range(polygon_count):
            polygon = _polygon_values(polygons, polygon_index)
            if not polygon or len(polygon) < 3:
                continue
            confidence = get_sequence_value(confidences, polygon_index)
            score = float(confidence) if confidence is not None else None
            spatial_iou = (
                _polygon_box_iou(polygon, requested_box, result)
                if requested_box is not None
                else None
            )
            if requested_box is not None and spatial_iou is not None and spatial_iou <= 0:
                continue
            mask_shape, mask_area = _mask_debug_values(masks, polygon_index)
            candidates.append(
                (
                    -(spatial_iou if spatial_iou is not None else -1.0),
                    score is None,
                    -(score or 0.0),
                    result_index,
                    polygon_index,
                    polygon,
                    score,
                    mask_shape,
                    mask_area,
                    spatial_iou,
                )
            )

    if not candidates:
        logger.debug("SAM3 box prompt returned no valid polygon candidates.")
        raise ValueError("SAM3 did not return a valid polygon for the selected box.")

    (
        _,
        _,
        _,
        result_index,
        polygon_index,
        polygon,
        confidence,
        mask_shape,
        mask_area,
        spatial_iou,
    ) = sorted(candidates)[0]
    logger.debug(
        "SAM3 box prompt selected result_index=%s polygon_index=%s confidence=%s "
        "box_iou=%s mask_shape=%s mask_area=%s polygon_point_count=%s",
        result_index,
        polygon_index,
        confidence,
        spatial_iou,
        mask_shape,
        mask_area,
        len(polygon),
    )
    return polygon, confidence


def annotations_from_sam3_result(result, prompts):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return []

    class_ids = getattr(boxes, "cls", None)
    confidences = getattr(boxes, "conf", None)
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xyn", None) if masks is not None else None

    annotations = []
    for object_index in range(len(xyxy)):
        box_xyxy = _xyxy_values(boxes, object_index)
        if box_xyxy is None:
            continue

        class_value = get_sequence_value(class_ids, object_index, 0)
        class_id = int(tensor_item(class_value, 0))
        confidence_value = get_sequence_value(confidences, object_index)
        confidence = tensor_item(confidence_value)

        annotations.append(
            Annotation(
                class_id=class_id,
                class_name=get_class_name(class_id, prompts),
                box_xyxy=box_xyxy,
                source=AnnotationSource.SAM3,
                confidence=None if confidence is None else float(confidence),
                polygon_xyn=_polygon_values(polygons, object_index),
                original_box_xyxy=box_xyxy,
                original_class_id=class_id,
                original_class_name=get_class_name(class_id, prompts),
            )
        )

    return annotations
