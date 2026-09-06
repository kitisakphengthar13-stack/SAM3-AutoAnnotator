def create_predictor(model_path, conf, half):
    # Importing Ultralytics is intentionally delayed until inference starts.
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(model_path),
        quantize=16 if bool(half) else 32,
        save=False,
    )
    return SAM3SemanticPredictor(overrides=overrides)


def create_box_segmenter(model_path):
    """Create the visual-prompt SAM interface for single-object box segmentation."""
    from ultralytics import SAM

    return SAM(str(model_path))
