"""TensorFlow / Keras chord classifier.

Loads the trained CNN once at construction time. The model is kept on
its default device (GPU if CUDA is visible, CPU otherwise - matches
the parent project's preference of "don't force TF onto CPU").

The classifier also performs the linear-interpolation stretch that the
offline pipeline uses to bring short (truncated) windows up to
``CQT_FEATURE_FRAMES`` columns.
"""

from __future__ import annotations

import gc
from typing import List

import numpy as np

from .. import config


def stretch_features_to_frames(
    feature_array: np.ndarray,
    target_frames: int = config.CQT_FEATURE_FRAMES,
) -> np.ndarray:
    """Linear-interpolate a 2-D CQT window to ``target_frames`` columns.

    Implementation is identical to the one in
    ``notebooks/onset/onset-classify_superflux.ipynb``. We keep a
    private copy here so the live demo does not depend on Jupyter
    path-machinery.
    """
    feature_array = np.asarray(feature_array)
    if feature_array.ndim != 2:
        raise ValueError(f"stretch_features_to_frames expects 2-D input, got shape {feature_array.shape}")

    source_frames = feature_array.shape[1]
    if source_frames == target_frames:
        return feature_array.astype(np.float32, copy=False)
    if source_frames == 1:
        return np.repeat(feature_array, target_frames, axis=1).astype(np.float32, copy=False)

    source_positions = np.linspace(0.0, 1.0, source_frames)
    target_positions = np.linspace(0.0, 1.0, target_frames)
    stretched_rows = [np.interp(target_positions, source_positions, row) for row in feature_array]
    return np.asarray(stretched_rows, dtype=np.float32)


class ChordClassifier:
    """Wraps the trained Keras CNN for inference.

    Parameters
    ----------
    model_path:
        Path to the ``.keras`` file. The file is loaded once at
        construction time; missing-file errors are surfaced eagerly so
        the WebSocket client gets a clean error rather than a series
        of confusing 500s.
    labels:
        List of class labels, indexed by ``argmax`` of the model
        output. Defaults to ``config.MODEL_LABELS``.
    """

    def __init__(self, model_path=config.MODEL_PATH, labels: List[str] = None) -> None:
        # Imported lazily so the rest of the package can be imported
        # even on machines where TensorFlow is not installed.
        import tensorflow as tf  # noqa: WPS433 (intentional local import)

        if labels is None:
            labels = list(config.MODEL_LABELS)

        self.labels: List[str] = list(labels)
        self.model = tf.keras.models.load_model(str(model_path))

        if self.model.output_shape[-1] != len(self.labels):
            raise ValueError(
                f"Model output size {self.model.output_shape[-1]} does not match "
                f"the number of labels ({len(self.labels)})."
            )

    def classify(self, cqt_window: np.ndarray) -> dict:
        """Run inference on a single CQT window.

        Parameters
        ----------
        cqt_window:
            2-D array of shape ``(CQT_FEATURE_BINS, source_frames)``
            with ``source_frames <= CQT_FEATURE_FRAMES``.

        Returns
        -------
        dict
            ``{raw_label, display_label, confidence, predicted_index,
                probabilities}``. ``display_label`` is the short
            ``"C:maj"``-style string; ``raw_label`` is the
            ``"C_major_4"``-style string the model was trained on.
        """
        stretched = stretch_features_to_frames(cqt_window, config.CQT_FEATURE_FRAMES)
        # The CNN expects ``(batch, bins, frames, 1)``.
        x = stretched[np.newaxis, ..., np.newaxis].astype(np.float32, copy=False)
        probabilities = self.model.predict(x, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        raw_label = self.labels[predicted_index]
        display_label = config.LABEL_DISPLAY_MAP.get(raw_label, raw_label)
        confidence = float(probabilities[predicted_index])

        # Release intermediate buffers eagerly to keep peak RAM low.
        del stretched, x
        gc.collect()

        return {
            "raw_label": raw_label,
            "display_label": display_label,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "probabilities": probabilities.astype(float).tolist(),
        }


__all__ = ["ChordClassifier", "stretch_features_to_frames"]
