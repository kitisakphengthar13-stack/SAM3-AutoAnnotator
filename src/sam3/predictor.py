def create_predictor(model_path, conf, half):
    # Importing Ultralytics is intentionally delayed until inference starts.
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(model_path),
        # Ultralytics 8.4 replaced the deprecated ``half`` flag with the
        # precision selector.  Be explicit in both modes so a cached predictor
        # cannot inherit a different precision from external configuration.
        quantize=16 if bool(half) else 32,
        save=False,
    )

    return SAM3SemanticPredictor(overrides=overrides)
