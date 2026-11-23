"""Métricas de sobreposição usadas no artigo (Dice, Jaccard, sensibilidade, especificidade, volume)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Overlap:
    dice: float
    jaccard: float
    sensitivity: float
    specificity: float
    volume_pred_ml: float
    volume_ref_ml: float
    volume_diff_pct: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _ratio(num: float, den: float) -> float:
    return num / den if den else 1.0


def compute_overlap(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, float, float]) -> Overlap:
    if pred.shape != ref.shape:
        raise ValueError(f"formas diferentes: {pred.shape} vs {ref.shape}")
    p = pred > 0
    r = ref > 0
    tp = float(np.count_nonzero(p & r))
    fp = float(np.count_nonzero(p & ~r))
    fn = float(np.count_nonzero(~p & r))
    tn = float(p.size) - tp - fp - fn
    voxel_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    vp = (tp + fp) * voxel_ml
    vr = (tp + fn) * voxel_ml
    return Overlap(
        dice=_ratio(2 * tp, 2 * tp + fp + fn),
        jaccard=_ratio(tp, tp + fp + fn),
        sensitivity=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
        volume_pred_ml=vp,
        volume_ref_ml=vr,
        volume_diff_pct=100.0 * (vp - vr) / vr if vr else 0.0,
    )
