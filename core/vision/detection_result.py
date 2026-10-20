"""Detection result value objects shared by every detector strategy."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hole:
    """One detected hole candidate, in pixel coordinates of the analysed image."""

    x_px: float
    y_px: float
    diameter_px: float
    circularity: float  # 4πA/P², 1.0 = perfect circle (1.0 where not applicable)
    confidence: float   # 0..1 composite score, strategy-specific


@dataclass(frozen=True)
class SearchRegion:
    """An area a two-stage strategy confined its hole search to — for the
    template strategies, one template match. Pixel coordinates of the analysed
    image, top-left corner plus size; ``score`` is how well the area matched."""

    x: int
    y: int
    width: int
    height: int
    score: float


@dataclass
class DetectionResult:
    """Outcome of one detector run on one image.

    ``holes`` is sorted by confidence, best first. A raised
    :class:`~core.utilities.exceptions.DetectionError` — not an empty result —
    signals an algorithm *failure*; an empty ``holes`` list is a legitimate
    "no hole present" answer.

    ``regions`` is empty for single-stage strategies. A two-stage one lists
    every area it searched, so the overlay can show *where* it looked — which
    is the only way to tell "the template was not found" from "the template
    was found but held no hole".
    """

    holes: list[Hole] = field(default_factory=list)
    processing_ms: float = 0.0
    regions: list[SearchRegion] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.holes)

    @property
    def best(self) -> Hole | None:
        return self.holes[0] if self.holes else None
