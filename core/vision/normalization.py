"""Optional image-normalization preprocessing step.

A single min-max contrast stretch (``cv2.normalize`` / ``NORM_MINMAX``) run
across the whole frame — gray or BGR alike — before a detector sees it, so a
frame shot under weaker/uneven lighting than the one the strategy was tuned
against fills the same 0-255 range it expects. Applied as one affine
transform over all channels together (not per-channel), so on a BGR frame it
rebalances brightness without shifting hue the way an independent per-channel
stretch would — important for ``dark_hole``'s channel-select mode.

Off by default and chosen per camera (``common.normalize_image`` in
detection.json, see ``VisionEngine``): a strategy's thresholds (e.g.
``opencv.detection_threshold``, ``dark_hole.min_contrast``) are tuned against
whatever contrast the live frames already have, so turning this on changes
what those thresholds mean and may need them re-tuned.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Return a min-max-stretched copy of *image*, mapped to the full 0-255 range.

    A flat (constant-value) frame has no range to stretch: left uncomputed,
    ``cv2.normalize`` would drive it entirely to 0 (black), which is a worse
    input to a detector than the original — so that case is returned as-is.
    """
    if image.min() == image.max():
        return image.copy()
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
