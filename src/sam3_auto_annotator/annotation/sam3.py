from sam3_auto_annotator.annotation.models import Annotation, AnnotationSource


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


def best_box_prompt_segmentation(results):
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
            candidates.append((score is None, -(score or 0.0), result_index, polygon_index, polygon, score))

    if not candidates:
        raise ValueError("SAM3 did not return a valid polygon for the selected box.")

    _, _, _, _, polygon, confidence = sorted(candidates)[0]
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
