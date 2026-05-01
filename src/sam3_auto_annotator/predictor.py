from ultralytics.models.sam import SAM3SemanticPredictor


def create_predictor(model_path, conf, half):
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(model_path),
        half=half,
        save=False,
    )

    return SAM3SemanticPredictor(overrides=overrides)

