from math import isfinite


def _finite_values(values, field_name):
    converted = [float(value) for value in values]
    if not all(isfinite(value) for value in converted):
        raise ValueError(f"{field_name} must contain only finite numbers.")
    return converted


def validate_image_size(image_width, image_height):
    width, height = _finite_values((image_width, image_height), "Image size")
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")


def validate_xyxy(box_xyxy):
    try:
        x1, y1, x2, y2 = _finite_values(box_xyxy, "xyxy box")
    except ValueError:
        raise
    except (TypeError, ValueError) as exc:
        raise ValueError("xyxy box must contain exactly four numeric values.") from exc
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy box with non-positive size: {box_xyxy}")
    return x1, y1, x2, y2


def clip_xyxy(box_xyxy, image_width, image_height):
    validate_image_size(image_width, image_height)
    x1, y1, x2, y2 = validate_xyxy(box_xyxy)
    clipped = (
        min(max(x1, 0.0), float(image_width)),
        min(max(y1, 0.0), float(image_height)),
        min(max(x2, 0.0), float(image_width)),
        min(max(y2, 0.0), float(image_height)),
    )
    return validate_xyxy(clipped)


def xyxy_to_xywh(box_xyxy):
    x1, y1, x2, y2 = validate_xyxy(box_xyxy)
    width = x2 - x1
    height = y2 - y1
    return x1 + width / 2.0, y1 + height / 2.0, width, height


def xyxy_to_yolo_xywhn(box_xyxy, image_width, image_height, clip=True):
    validate_image_size(image_width, image_height)
    box = clip_xyxy(box_xyxy, image_width, image_height) if clip else validate_xyxy(box_xyxy)
    x_center, y_center, width, height = xyxy_to_xywh(box)
    return (
        x_center / float(image_width),
        y_center / float(image_height),
        width / float(image_width),
        height / float(image_height),
    )
