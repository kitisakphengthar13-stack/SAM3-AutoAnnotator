import matplotlib.pyplot as plt


def prediction_image_path(prediction_results_dir, image_path):
    return prediction_results_dir / f"{image_path.stem}_predicted.png"


def save_or_show_prediction(result, image_path, prediction_results_dir=None, show=False):
    if prediction_results_dir is None and not show:
        return

    prediction = result.plot()
    if prediction_results_dir is not None:
        prediction_results_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(prediction_image_path(prediction_results_dir, image_path), prediction)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(prediction)
        plt.axis("off")
        plt.show()

