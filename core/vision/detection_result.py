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


@dataclass
class DetectionResult:
    """Outcome of one detector run on one image.

    ``holes`` is sorted by confidence, best first. A raised
    :class:`~core.utilities.exceptions.DetectionError` — not an empty result —
    signals an algorithm *failure*; an empty ``holes`` list is a legitimate
    "no hole present" answer.
    """

    holes: list[Hole] = field(default_factory=list)
    processing_ms: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.holes)

    @property
    def best(self) -> Hole | None:
        return self.holes[0] if self.holes else None
